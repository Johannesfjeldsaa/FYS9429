"""
ae_experiment_runner.py
=======================
Configurable autoencoder factory, experiment registry, sweep runner, and
metric functions for the AE-VARX hyperparameter sweep.

Architecture overview
---------------------
``ConfigurableMLP`` – symmetric residual MLP with optional LayerNorm, Dropout
    and skip connections.  Inherits from
    ``meteor.noise_model.ae_construct.BaseAutoencoder``.

``make_configurable_ae(...)`` → ``AEConfig``
    Factory that closes over architecture hyper-parameters and returns an
    ``AEConfig`` ready to train.

``ExperimentRegistry``
    JSON-backed persistent registry for completed runs (skip/resume support).

``run_ae_experiment(config_dict, X_ae_mlp, ...)`` → ``dict``
    Train one AE + VARX configuration and return a merged metrics dict.

``run_sweep(grid, ...)`` → ``pd.DataFrame``
    Cartesian-product sweep over ``grid``; saves ``experiment_registry.json``
    and ``sweep_results.parquet`` into ``results_dir``.

Metric helpers
--------------
``compute_reconstruction_metrics`` – RMSE, EV, spatial corr, regional RMSE,
    skewness/kurtosis MAE.
``compute_latent_diagnostics`` – effective rank, ACF, off-diagonal correlation.
``compute_varx_diagnostics`` – spectral radius, AIC/BIC, Ljung-Box, KS normality.
``compute_stochastic_rollout_stability`` – NaN/Inf fraction, variance/mean drift.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis, kstest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR

import pandas as pd
import torch
import torch.nn as nn
_TORCH = True

from meteor.noise_model.ae_construct import AEConfig, BaseAutoencoder
from meteor.noise_model.noise_model_data_utils import (
    build_exog_matrix,
    create_harmonic_features,
)

__all__ = [
    "ConfigurableMLP",
    "make_configurable_ae",
    "ExperimentRegistry",
    "run_ae_experiment",
    "run_sweep",
    "compute_reconstruction_metrics",
    "compute_latent_diagnostics",
    "compute_varx_diagnostics",
    "compute_stochastic_rollout_stability",
]

# ---------------------------------------------------------------------------
# Configurable MLP autoencoder
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Linear + (optional LayerNorm) + activation + (optional skip) + Dropout."""

    def __init__(
        self,
        dim: int,
        activation,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.norm   = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.linear = nn.Linear(dim, dim)
        self.act    = activation()
        self.drop   = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.drop(self.act(self.norm(self.linear(x))))
        return x + out if self.use_residual else out


class _MLP(nn.Module):
    """
    One half (encoder or decoder) of a symmetric MLP.

    Per hidden dimension: Linear(prev→h) → activation → _ResidualBlock(h)
    Final: Linear(last_h→out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple[int, ...],
        activation,
        dropout: float,
        use_layernorm: bool,
        use_residual: bool,
    ):
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            layers.append(
                _ResidualBlock(h, activation, dropout, use_layernorm, use_residual)
            )
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConfigurableMLP(BaseAutoencoder):
    """
    Configurable symmetric MLP autoencoder.

    Supports residual skip connections, LayerNorm, Dropout and custom
    activations.  Inherits from ``BaseAutoencoder`` (numpy I/O at predict
    time, CPU-at-rest contract).

    Parameters
    ----------
    input_dim : int
        Flattened spatial dimension (n_lat × n_lon).
    latent_dim : int
        Bottleneck size.
    hidden : tuple[int, ...]
        Hidden widths in *encoder* order; mirrored in decoder.
    activation : nn.Module subclass (not instance)
        Default ``nn.Hardswish``.
    dropout : float
        Dropout probability; 0 disables.
    use_layernorm : bool
        Apply LayerNorm inside each residual block.
    use_residual : bool
        Add skip connections inside each residual block.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: tuple[int, ...] = (1024, 512, 256),
        activation=None,
        dropout: float = 0.30,
        use_layernorm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        if activation is None:
            activation = nn.Hardswish
        self.encoder = _MLP(
            input_dim, latent_dim, hidden,
            activation, dropout, use_layernorm, use_residual,
        )
        self.decoder = _MLP(
            latent_dim, input_dim, tuple(reversed(hidden)),
            activation, dropout, use_layernorm, use_residual,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_configurable_ae(
    latent_dim: int,
    hidden: tuple[int, ...] = (1024, 512, 256),
    activation=None,
    dropout: float = 0.30,
    use_residual: bool = True,
    use_layernorm: bool = True,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "auto",
    verbose: bool = False,
    optimizer: str = "adamw",
    loss: str = "mse",
) -> AEConfig:
    """
    Build an ``AEConfig`` that trains a :class:`ConfigurableMLP`.

    The factory closes over the architecture hyper-parameters and produces a
    fresh model when called with the input array (``input_dim`` is inferred
    from ``X_np.shape[1]`` at training time).

    Parameters
    ----------
    latent_dim : int
    hidden : tuple[int, ...]
    activation : nn.Module subclass or None  (default ``nn.Hardswish``)
    dropout, use_residual, use_layernorm : architecture toggles
    epochs, batch_size, lr, weight_decay, device, verbose : training config

    Returns
    -------
    AEConfig
    """
    if activation is None:
        activation = nn.Hardswish

    def _factory(X_np: np.ndarray) -> ConfigurableMLP:
        return ConfigurableMLP(
            input_dim=X_np.shape[1],
            latent_dim=latent_dim,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
            use_residual=use_residual,
        )

    _opt_cls = _OPT_MAP.get(optimizer, torch.optim.AdamW)

    def _opt_factory(model):
        kw = {"lr": lr, "weight_decay": weight_decay}
        if optimizer == "sgd":
            kw["momentum"] = 0.9
        return _opt_cls(model.parameters(), **kw)

    return AEConfig(
        model_factory=_factory,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
        optimizer_factory=_opt_factory,
        loss_fn=_LOSS_MAP.get(loss, nn.MSELoss)(),
        arch_meta={
            "latent_dim": latent_dim,
            "hidden": list(hidden),
            "activation": getattr(activation, "__name__", str(activation)),
            "dropout": dropout,
            "use_residual": use_residual,
            "use_layernorm": use_layernorm,
            "optimizer": optimizer,
            "loss": loss,
        },
    )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """
    Persistent JSON registry of AE sweep experiments.

    Provides content-hash based run IDs for skip/resume support.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file; created (with parent dirs) if absent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path) as fh:
                self._data: dict = json.load(fh)
        else:
            self._data = {}
            self._save()

    def config_id(self, config_dict: dict) -> str:
        """Deterministic 12-char hex ID for a config dict."""
        canonical = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def is_done(self, config_dict: dict) -> bool:
        return self.config_id(config_dict) in self._data

    def register(self, config_dict: dict, metrics: dict, run_id: str | None = None):
        """Persist a completed run."""
        cid = self.config_id(config_dict)
        self._data[cid] = {
            "run_id":    run_id or str(uuid.uuid4())[:8],
            "config":    config_dict,
            "metrics":   metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def to_dataframe(self):
        """Return all results as a pandas DataFrame (one row per run)."""
        rows = []
        for entry in self._data.values():
            row = {
                **entry["config"],
                **entry["metrics"],
                "run_id":    entry["run_id"],
                "timestamp": entry["timestamp"],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _save(self):
        with open(self.path, "w") as fh:
            json.dump(self._data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_reconstruction_metrics(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    lats: np.ndarray,
) -> dict:
    """
    Reconstruction quality metrics comparing original and reconstructed fields.

    Parameters
    ----------
    X_orig, X_recon : np.ndarray  shape ``(T, n_lat * n_lon)``
    lats : np.ndarray  1-D latitude array (same order as spatial layout)

    Returns
    -------
    dict with keys:
        rmse, explained_variance, spatial_corr,
        arctic_rmse, sh_rmse, skew_mae, kurt_mae.
    """
    n_lon = X_orig.shape[1] // len(lats)
    cos_w = np.cos(np.deg2rad(lats))
    cos_w_flat = np.repeat(cos_w, n_lon)
    cos_w_flat = cos_w_flat / cos_w_flat.sum()

    diff = X_recon - X_orig
    rmse = float(np.sqrt((diff ** 2 * cos_w_flat).sum(axis=1).mean()))

    ss_res = float((diff ** 2).mean())
    ss_tot = float((X_orig - X_orig.mean()).var())
    expl_var = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

    orig_c  = X_orig  - X_orig.mean(axis=1,  keepdims=True)
    recon_c = X_recon - X_recon.mean(axis=1, keepdims=True)
    num  = (orig_c * recon_c * cos_w_flat).sum(axis=1)
    denom = (
        np.sqrt((orig_c  ** 2 * cos_w_flat).sum(axis=1)) *
        np.sqrt((recon_c ** 2 * cos_w_flat).sum(axis=1))
    )
    spatial_corr = float(np.where(denom > 0, num / denom, 0.0).mean())

    arctic_mask = np.repeat(lats >= 66,  n_lon)
    sh_mask     = np.repeat(lats <= -30, n_lon)
    arctic_rmse = float(np.sqrt((diff[:, arctic_mask] ** 2).mean()))
    sh_rmse     = float(np.sqrt((diff[:, sh_mask]     ** 2).mean()))

    n_lat = len(lats)
    X_o_3d = X_orig.reshape(-1,  n_lat, n_lon)
    X_r_3d = X_recon.reshape(-1, n_lat, n_lon)
    skew_mae = float(np.abs(skew(X_o_3d, axis=0) - skew(X_r_3d, axis=0)).mean())
    kurt_mae = float(
        np.abs(
            kurtosis(X_o_3d, axis=0, fisher=True) -
            kurtosis(X_r_3d, axis=0, fisher=True)
        ).mean()
    )

    return dict(
        rmse=rmse,
        explained_variance=expl_var,
        spatial_corr=spatial_corr,
        arctic_rmse=arctic_rmse,
        sh_rmse=sh_rmse,
        skew_mae=skew_mae,
        kurt_mae=kurt_mae,
    )


def compute_latent_diagnostics(pcs: np.ndarray) -> dict:
    """
    Diagnostics for the latent code time series.

    Parameters
    ----------
    pcs : np.ndarray  shape ``(T, latent_dim)``

    Returns
    -------
    dict with keys:
        effective_rank, mean_skew, mean_kurt,
        mean_acf_lag1, mean_acf_lag12,
        corr_matrix_offdiag_mean.
    """
    n_t, n_dim = pcs.shape

    # Roy & Vetterli (2007) effective rank
    cov = np.cov(pcs, rowvar=False)
    eigs = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    p = eigs / (eigs.sum() + 1e-12)
    p = p[p > 0]
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))

    mean_skew = float(skew(pcs, axis=0).mean())
    mean_kurt = float(kurtosis(pcs, axis=0, fisher=True).mean())

    def _acf_lag(x, lag):
        x = x - x.mean()
        c0 = float(np.dot(x, x) / len(x))
        if c0 == 0 or lag >= len(x):
            return 0.0
        return float(np.dot(x[:-lag], x[lag:]) / (len(x) * c0))

    acf1  = np.array([_acf_lag(pcs[:, d], 1)  for d in range(n_dim)])
    acf12 = np.array([_acf_lag(pcs[:, d], 12) for d in range(n_dim)])

    corr = np.corrcoef(pcs, rowvar=False)
    mask = ~np.eye(n_dim, dtype=bool)
    offdiag_mean = float(np.abs(corr[mask]).mean()) if n_dim > 1 else 0.0

    return dict(
        effective_rank=effective_rank,
        mean_skew=mean_skew,
        mean_kurt=mean_kurt,
        mean_acf_lag1=float(acf1.mean()),
        mean_acf_lag12=float(acf12.mean()),
        corr_matrix_offdiag_mean=offdiag_mean,
    )


def compute_varx_diagnostics(ae_noise_model) -> dict:
    """
    Diagnostics for the VARX model fitted inside an ``AEVARXNoiseModel``.

    Uses the companion-matrix spectral radius calculation from cell 40 of
    ``METEOR_AE-VARX_tas.ipynb``.

    Parameters
    ----------
    ae_noise_model : AEVARXNoiseModel (fitted)

    Returns
    -------
    dict with keys:
        spectral_radius, aic, bic,
        ljungbox_mean_pval, ks_normality_mean_pval.
    """
    vr = ae_noise_model.varx_results
    var_params = vr.params
    latent_dim = var_params.shape[1]
    lag_order  = ae_noise_model.lag_order

    # Companion matrix spectral radius
    C = np.zeros((lag_order * latent_dim, lag_order * latent_dim))
    for lag_i in range(lag_order):
        start = 1 + lag_i * latent_dim
        A = var_params[start: start + latent_dim, :].T
        C[:latent_dim, lag_i * latent_dim: (lag_i + 1) * latent_dim] = A
    if lag_order > 1:
        C[latent_dim:, : latent_dim * (lag_order - 1)] = np.eye(
            latent_dim * (lag_order - 1)
        )
    sr = float(np.max(np.abs(np.linalg.eigvals(C))))

    aic = float(getattr(vr, "aic", np.nan))
    bic = float(getattr(vr, "bic", np.nan))

    resid = np.asarray(vr.resid) if hasattr(vr, "resid") else None

    # Ljung-Box on first 5 latent dims, lag 12
    lb_pvals: list[float] = []
    if resid is not None:
        for d in range(min(5, resid.shape[1])):
            try:
                res = acorr_ljungbox(resid[:, d], lags=[12], return_df=True)
                lb_pvals.append(float(res["lb_pvalue"].iloc[0]))
            except Exception:
                pass
    lb_mean = float(np.mean(lb_pvals)) if lb_pvals else np.nan

    # KS normality on first 5 latent dims
    ks_pvals: list[float] = []
    if resid is not None:
        for d in range(min(5, resid.shape[1])):
            z = resid[:, d]
            z = (z - z.mean()) / (z.std() + 1e-12)
            _, p = kstest(z, "norm")
            ks_pvals.append(p)
    ks_mean = float(np.mean(ks_pvals)) if ks_pvals else np.nan

    return dict(
        spectral_radius=sr,
        aic=aic,
        bic=bic,
        ljungbox_mean_pval=lb_mean,
        ks_normality_mean_pval=ks_mean,
    )


def compute_stochastic_rollout_stability(
    noise_model,
    t_glob: np.ndarray,
    n_real: int = 20,
) -> dict:
    """
    Test stochastic rollout stability over the scenario period (typically
    2015–2100, 85 years × 12 months = 1020 time steps).

    Parameters
    ----------
    noise_model : AEVARXNoiseModel (fitted)
    t_glob : np.ndarray  global-mean temperature trajectory (length T_scenario)
    n_real : int  number of realisations

    Returns
    -------
    dict with keys:
        nan_fraction, inf_fraction,
        variance_drift_ratio, mean_drift_abs.
    """
    n_time = len(t_glob)
    X_exog = build_exog_matrix(
        create_harmonic_features(np.arange(n_time), t_glob),
        noise_model.use_exog,
    )

    all_pcs: list[np.ndarray] = []
    for _ in range(n_real):
        try:
            pcs = noise_model._generate_stochastic_pcs(X_exog, n_time)
            all_pcs.append(pcs)
        except Exception:
            pass

    if not all_pcs:
        return dict(
            nan_fraction=1.0, inf_fraction=1.0,
            variance_drift_ratio=np.nan, mean_drift_abs=np.nan,
        )

    arr = np.stack(all_pcs, axis=0)  # (n_real, T, latent_dim)
    nan_frac = float(np.isnan(arr).mean())
    inf_frac = float(np.isinf(arr).mean())

    var_start = float(np.nanvar(arr[:, :12, :]))
    var_end   = float(np.nanvar(arr[:, -12:, :]))
    var_drift = var_end / (var_start + 1e-12)

    mean_traj  = np.nanmean(arr, axis=(0, 2))  # (T,)
    mean_drift = float(np.abs(mean_traj[-12:].mean() - mean_traj[:12].mean()))

    return dict(
        nan_fraction=nan_frac,
        inf_fraction=inf_frac,
        variance_drift_ratio=var_drift,
        mean_drift_abs=mean_drift,
    )


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

_ACT_MAP: dict = {}
_OPT_MAP: dict = {}
_LOSS_MAP: dict = {}
if _TORCH:
    _ACT_MAP = {
        "Hardswish": nn.Hardswish,
        "ReLU":      nn.ReLU,
        "GELU":      nn.GELU,
        "SiLU":      nn.SiLU,
    }
    _OPT_MAP = {
        "adamw": torch.optim.AdamW,
        "adam":  torch.optim.Adam,
        "sgd":   torch.optim.SGD,
    }
    _LOSS_MAP = {
        "mse":   nn.MSELoss,
        "mae":   nn.L1Loss,
        "huber": nn.HuberLoss,
    }


def run_ae_experiment(
    config_dict: dict,
    X_ae_mlp: np.ndarray,
    X_ae_conv: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    t_glob: np.ndarray,
    lag_order: int = 2,
    use_exog: str = "temp_only",
    checkpoint_dir: str | Path | None = None,
    verbose: bool = False,
    resume: bool = False,
) -> dict:
    """
    Train one AE + VARX configuration and return a metrics dict.

    The VARX fitting is performed directly via statsmodels (no call to
    ``AEVARXNoiseModel.fit()`` — see implementation notes in the session
    summary for why).

    Parameters
    ----------
    config_dict : dict
        Keys: ``latent_dim``, ``hidden``, ``activation`` (str),
        ``dropout``, ``use_residual``, ``use_layernorm``,
        ``epochs``, ``batch_size``, ``lr``, ``weight_decay``, ``device``.
    X_ae_mlp : np.ndarray  shape ``(T, n_lat * n_lon)``
    X_ae_conv : np.ndarray  shape ``(T, 1, n_lat, n_lon)``
    lats, lons : np.ndarray
    t_glob : np.ndarray  global-mean temperature trajectory (length T)
    lag_order : int  VAR lag order
    use_exog : str  e.g. ``"temp_only"`` or ``"all"``
    checkpoint_dir : str or Path or None
    verbose : bool
    resume : bool
        If ``True`` **and** a checkpoint + sidecar ``_meta.json`` exist for
        this config, load the saved weights and train only the remaining
        epochs (``config_dict["epochs"] - epochs_already_done``).  If no
        checkpoint is found the model is trained from scratch as normal.
        A ``_meta.json`` sidecar recording ``epochs_done`` is always written
        alongside the ``.pt`` file when ``checkpoint_dir`` is set.

    Returns
    -------
    dict  merged reconstruction + latent + VARX + rollout metrics + train_time_s
    """
    act_key = config_dict.get("activation", "Hardswish")
    activation = _ACT_MAP.get(act_key, nn.Hardswish) if _TORCH else None

    ae_cfg = make_configurable_ae(
        latent_dim    =config_dict["latent_dim"],
        hidden        =tuple(config_dict.get("hidden", [1024, 512, 256])),
        activation    =activation,
        dropout       =config_dict.get("dropout", 0.30),
        use_residual  =config_dict.get("use_residual", True),
        use_layernorm =config_dict.get("use_layernorm", True),
        epochs        =config_dict.get("epochs", 100),
        batch_size    =config_dict.get("batch_size", 64),
        lr            =config_dict.get("lr", 1e-3),
        weight_decay  =config_dict.get("weight_decay", 1e-5),
        device        =config_dict.get("device", "auto"),
        verbose       =verbose,
        optimizer     =config_dict.get("optimizer", "adamw"),
        loss          =config_dict.get("loss", "mse"),
    )

    # --- Train AE (with optional checkpoint resume) ------------------------
    target_epochs = config_dict.get("epochs", 100)
    epochs_already_done = 0
    ckpt_path = None
    ckpt_meta_path = None

    if checkpoint_dir is not None and _TORCH:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cid = ExperimentRegistry.__new__(ExperimentRegistry).config_id(config_dict)
        ckpt_path      = ckpt_dir / f"ae_{cid}.pt"
        ckpt_meta_path = ckpt_dir / f"ae_{cid}_meta.json"

        if resume and ckpt_path.exists() and ckpt_meta_path.exists():
            with open(ckpt_meta_path) as _mf:
                epochs_already_done = json.load(_mf).get("epochs_done", 0)
            if verbose:
                print(f"  [resume] checkpoint found — "
                      f"{epochs_already_done}/{target_epochs} epochs done, "
                      f"training {max(0, target_epochs - epochs_already_done)} more")

    remaining_epochs = max(0, target_epochs - epochs_already_done)

    t0 = time.time()
    if remaining_epochs == 0:
        # Already fully trained — just reload weights, skip training
        ae = ae_cfg.model_factory(X_ae_mlp)
        ae.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True))
        ae.eval()
        train_time = 0.0
    elif epochs_already_done > 0 and ckpt_path is not None and ckpt_path.exists():
        # Partial resume: load weights, run remaining epochs with a fresh optimiser
        ae = ae_cfg.model_factory(X_ae_mlp)
        ae.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True))
        optimizer = (
            ae_cfg.optimizer_factory(ae)
            if ae_cfg.optimizer_factory is not None
            else torch.optim.Adam(ae.parameters(), lr=ae_cfg.lr)
        )
        loss_fn = ae_cfg.loss_fn if ae_cfg.loss_fn is not None else nn.MSELoss()
        ae.fit(
            X_ae_mlp,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=remaining_epochs,
            batch_size=ae_cfg.batch_size,
            lr=ae_cfg.lr,
            device=ae_cfg.device,
            verbose=verbose,
        )
        train_time = time.time() - t0
    else:
        # Train from scratch
        ae = ae_cfg.get_trained_ae(X_ae_mlp)
        train_time = time.time() - t0

    if checkpoint_dir is not None and _TORCH:
        torch.save(ae.state_dict(), ckpt_path)
        with open(ckpt_meta_path, "w") as _mf:
            json.dump({"epochs_done": epochs_already_done + remaining_epochs}, _mf)

    # --- Reconstruction metrics (MLP flat) ----------------------------------
    pcs      = ae.encode(X_ae_mlp)
    X_recon  = ae.decode(pcs)
    recon_metrics  = compute_reconstruction_metrics(X_ae_mlp, X_recon, lats)
    latent_metrics = compute_latent_diagnostics(pcs)

    # --- Fit VARX on latent codes -------------------------------------------
    varx_metrics   = dict(spectral_radius=np.nan, aic=np.nan, bic=np.nan,
                          ljungbox_mean_pval=np.nan, ks_normality_mean_pval=np.nan)
    rollout_metrics = dict(nan_fraction=np.nan, inf_fraction=np.nan,
                           variance_drift_ratio=np.nan, mean_drift_abs=np.nan)
    try:
        n_time = len(t_glob)
        X_exog = build_exog_matrix(
            create_harmonic_features(np.arange(n_time), t_glob),
            use_exog,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            varx_results = VAR(endog=pcs, exog=X_exog).fit(lag_order)

        # Construct minimal noise-model-like object for diagnostics
        from meteor.noise_model.ae_varx import AEVARXNoiseModel as _AEVARXRef
        import types
        nm = types.SimpleNamespace(
            varx_results=varx_results,
            lag_order=lag_order,
            use_exog=use_exog,
            ae=ae,
            fitted=True,
            noise_pc_distribution="normal",
            t_df=None,
        )
        nm._generate_stochastic_pcs = types.MethodType(
            _AEVARXRef._generate_stochastic_pcs, nm
        )

        varx_metrics   = compute_varx_diagnostics(nm)
        rollout_metrics = compute_stochastic_rollout_stability(nm, t_glob)
    except Exception as e:
        if verbose:
            print(f"  VARX fitting/diagnostics failed: {e}")

    return {
        **recon_metrics,
        **latent_metrics,
        **varx_metrics,
        **rollout_metrics,
        "train_time_s": train_time,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    grid: dict,
    X_ae_mlp: np.ndarray,
    X_ae_conv: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    t_glob: np.ndarray,
    results_dir: str | Path = "results",
    lag_order: int = 2,
    use_exog: str = "temp_only",
    skip_done: bool = True,
    verbose: bool = True,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Run a full Cartesian-product hyperparameter sweep.

    Parameters
    ----------
    grid : dict
        Keys are hyper-parameter names, values are lists.  Example::

            grid = {
                "latent_dim":   [16, 32, 64, 128],
                "hidden":       [(512,), (1024, 512), (1024, 512, 256)],
                "dropout":      [0.1, 0.3],
                "use_residual": [True, False],
            }

    X_ae_mlp  : np.ndarray  (T, n_lat * n_lon)
    X_ae_conv : np.ndarray  (T, 1, n_lat, n_lon)
    lats, lons : np.ndarray
    t_glob : np.ndarray  global temperature trajectory
    results_dir : str or Path
    lag_order, use_exog : VARX config
    skip_done : bool  skip configurations already in registry
    verbose : bool

    Returns
    -------
    pandas.DataFrame  all results (existing + newly computed)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(results_dir / "experiment_registry.json")

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)

    for idx, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        # Ensure hidden is JSON-serialisable (list, not tuple)
        if "hidden" in config and not isinstance(config["hidden"], list):
            config["hidden"] = list(config["hidden"])

        if skip_done and registry.is_done(config):
            if verbose:
                print(f"[{idx+1}/{total}] SKIP  {config}")
            continue

        if verbose:
            print(f"[{idx+1}/{total}] RUN   {config}", flush=True)

        try:
            metrics = run_ae_experiment(
                config_dict   =config,
                X_ae_mlp      =X_ae_mlp,
                X_ae_conv     =X_ae_conv,
                lats          =lats,
                lons          =lons,
                t_glob        =t_glob,
                lag_order     =lag_order,
                use_exog      =use_exog,
                checkpoint_dir=results_dir / "checkpoints",
                verbose       =False,  # suppress epoch-level output; summary printed below
                resume        =resume,
            )
            registry.register(config, metrics)
            if verbose:
                ld = config.get('latent_dim', '?')
                hid = config.get('hidden', '?')
                do = config.get('dropout', '?')
                res = config.get('use_residual', '?')
                print(
                    f"       latent={ld}  hidden={hid}  dropout={do}  residual={res}"
                    f"  RMSE={metrics.get('rmse', float('nan')):.4f}"
                    f"  SR={metrics.get('spectral_radius', float('nan')):.4f}"
                    f"  t={metrics.get('train_time_s', float('nan')):.0f}s"
                )
        except Exception as e:
            if verbose:
                print(f"       ERROR: {e}")

    df = registry.to_dataframe()
    df.to_parquet(results_dir / "sweep_results.parquet")
    return df


# ---------------------------------------------------------------------------
# Convenience: build AEConfig directly from a champion-style config dict
# ---------------------------------------------------------------------------

def make_configurable_ae_from_dict(d: dict) -> "AEConfig":
    """
    Build an :class:`AEConfig` from a flat config dict (e.g. a loaded
    champion JSON).

    Recognised keys mirror those of :func:`make_configurable_ae`:
    ``latent_dim``, ``hidden``, ``activation``, ``dropout``,
    ``use_residual``, ``use_layernorm``, ``epochs``, ``batch_size``,
    ``lr``, ``weight_decay``, ``device``, ``verbose``,
    ``optimizer``, ``loss``.

    Unknown keys (e.g. ``variable``, ``model``, ``rmse_p2_baseline``) are
    silently ignored.

    Parameters
    ----------
    d : dict

    Returns
    -------
    AEConfig
    """
    act_key    = d.get("activation", "Hardswish")
    activation = _ACT_MAP.get(act_key, nn.Hardswish) if _TORCH else None
    return make_configurable_ae(
        latent_dim   =d["latent_dim"],
        hidden       =tuple(d.get("hidden",       [1024, 512, 256])),
        activation   =activation,
        dropout      =d.get("dropout",      0.30),
        use_residual =d.get("use_residual", True),
        use_layernorm=d.get("use_layernorm", True),
        epochs       =d.get("epochs",       100),
        batch_size   =d.get("batch_size",   64),
        lr           =d.get("lr",           1e-3),
        weight_decay =d.get("weight_decay", 1e-5),
        device       =d.get("device",       "auto"),
        verbose      =d.get("verbose",      False),
        optimizer    =d.get("optimizer",    "adamw"),
        loss         =d.get("loss",         "mse"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main_cli() -> None:
    """
    Command-line interface for running a single AE experiment from a config
    file.

    Usage
    -----
    ::

        python ae_experiment_runner.py \\
            --config /scratch/johannlf/ae_sweep/pr/champion_p6_pr.json \\
            --data-dir /scratch/johannlf/ae_sweep/pr \\
            [--checkpoint-dir /scratch/johannlf/ae_sweep/pr/checkpoints] \\
            [--resume] \\
            [--verbose]

    Expected files in ``--data-dir``
    ---------------------------------
    ``X_ae_mlp.npy``   – shape (T, n_lat*n_lon), float32  
    ``X_ae_conv.npy``  – shape (T, 1, n_lat, n_lon), float32  
    ``lats.npy``       – 1-D latitude array  
    ``lons.npy``       – 1-D longitude array  
    ``t_glob.npy``     – 1-D global-mean temperature trajectory  

    Save from the notebook once::

        np.save(SCRATCH_DIR / "X_ae_mlp.npy",  X_ae_mlp)
        np.save(SCRATCH_DIR / "X_ae_conv.npy", X_ae_conv)
        np.save(SCRATCH_DIR / "lats.npy",  lats)
        np.save(SCRATCH_DIR / "lons.npy",  lons)
        np.save(SCRATCH_DIR / "t_glob.npy", t_glob)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single AE-VARX experiment from a JSON config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a JSON config file (champion format or custom).",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing X_ae_mlp.npy, lats.npy, lons.npy, t_glob.npy.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory to save/load .pt checkpoints. "
             "Defaults to <data-dir>/checkpoints.",
    )
    parser.add_argument(
        "--lag-order", type=int, default=2,
        help="VARX lag order.",
    )
    parser.add_argument(
        "--use-exog", default="none",
        choices=["none", "temp_only", "all"],
        help="Exogenous regressors for VARX.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from an existing checkpoint if available.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch training loss.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else data_dir / "checkpoints"

    # Load config
    with open(args.config) as fh:
        config_dict = json.load(fh)

    # Load pre-saved numpy arrays
    print(f"Loading data from {data_dir} …")
    X_ae_mlp  = np.load(data_dir / "X_ae_mlp.npy")
    X_ae_conv = np.load(data_dir / "X_ae_conv.npy")
    lats      = np.load(data_dir / "lats.npy")
    lons      = np.load(data_dir / "lons.npy")
    t_glob    = np.load(data_dir / "t_glob.npy")
    print(f"  X_ae_mlp : {X_ae_mlp.shape}   X_ae_conv : {X_ae_conv.shape}")
    print(f"  lats     : {lats.shape}        t_glob    : {t_glob.shape}")

    print(f"\nConfig: {json.dumps(config_dict, indent=2, default=str)}\n")

    metrics = run_ae_experiment(
        config_dict   =config_dict,
        X_ae_mlp      =X_ae_mlp,
        X_ae_conv     =X_ae_conv,
        lats          =lats,
        lons          =lons,
        t_glob        =t_glob,
        lag_order     =args.lag_order,
        use_exog      =args.use_exog,
        checkpoint_dir=ckpt_dir,
        verbose       =args.verbose,
        resume        =args.resume,
    )

    print("\nMetrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<32s} {v}")


if __name__ == "__main__":
    _main_cli()
