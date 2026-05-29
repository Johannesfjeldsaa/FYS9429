"""
ae_diagnostics.py
=================
Reusable diagnostic helpers for anomaly fields and AE reconstructions.

Extracted and generalised from ``investigating_anomalies.ipynb`` (cells 1–31).
All functions accept ``X_field`` shaped as ``(T, 1, n_lat, n_lon)`` (conv
layout) or ``(T, n_lat, n_lon)`` (no channel dim).  Lats are expected in
south-to-north order (standard CMIP6 convention).

Public API
----------
plot_sample_on_ax
plot_monthly_samples_grid
seasonal_cycle_before_after
summarize_anomaly_field_core
compare_field_statistics
qqplot_anomaly_field
compute_eofs
plot_eof_maps
compute_temporal_stats_from_flat
plot_temporal_structure_flat
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import skew, kurtosis, probplot
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

__all__ = [
    "plot_sample_on_ax",
    "plot_monthly_samples_grid",
    "seasonal_cycle_before_after",
    "summarize_anomaly_field_core",
    "compare_field_statistics",
    "qqplot_anomaly_field",
    "compute_eofs",
    "plot_eof_maps",
    "compute_temporal_stats_from_flat",
    "plot_temporal_structure_flat",
]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_conv(X_field: np.ndarray) -> np.ndarray:
    """Ensure ``X_field`` has shape ``(T, 1, n_lat, n_lon)``."""
    if X_field.ndim == 3:
        return X_field[:, np.newaxis, :, :]
    if X_field.ndim == 4 and X_field.shape[1] == 1:
        return X_field
    raise ValueError(
        f"Expected (T, n_lat, n_lon) or (T, 1, n_lat, n_lon), got {X_field.shape}."
    )


# ---------------------------------------------------------------------------
# Spatial sample plots
# ---------------------------------------------------------------------------

def plot_sample_on_ax(
    ax,
    sample_2d: np.ndarray,
    extent: list[float],
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    add_coastlines: bool = True,
):
    """
    Plot a single 2-D climate field on a Cartopy axis.

    Parameters
    ----------
    ax : GeoAxesSubplot
    sample_2d : np.ndarray  shape ``(n_lat, n_lon)``
    extent : list[float]  ``[lon_min, lon_max, lat_min, lat_max]``
    title, vmin, vmax, cmap : as named
    add_coastlines : bool

    Returns
    -------
    matplotlib.image.AxesImage
    """
    im = ax.imshow(
        sample_2d,
        origin="lower",
        transform=ccrs.PlateCarree(),
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    if add_coastlines:
        ax.add_feature(
            cfeature.LAND, facecolor="none", edgecolor="black", lw=0.5, zorder=10
        )
    return im


def plot_monthly_samples_grid(
    X: np.ndarray,
    random_indices: dict,
    extent: list[float],
    month_names: list[str] | None = None,
    cmap: str = "coolwarm",
    suptitle: str = "X_ae samples by month",
) -> plt.Figure:
    """
    Plot several random samples per month in a (months × samples) grid.

    Parameters
    ----------
    X : np.ndarray  shape ``(n_samples, 1, n_lat, n_lon)``
    random_indices : dict[int, array-like]  keys 1–12
    extent : list[float]
    month_names, cmap, suptitle : as named

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = _to_conv(X)
    months = sorted(random_indices.keys())
    n_months = len(months)
    n_per_month = len(next(iter(random_indices.values())))

    all_vals = np.concatenate(
        [X[random_indices[m], 0, :, :] for m in months], axis=0
    )
    cmap_max = max(abs(all_vals.min()), abs(all_vals.max()))
    vmin, vmax = -cmap_max, cmap_max

    fig = plt.figure(figsize=(4 * n_per_month, 2.5 * n_months))
    gs = GridSpec(nrows=n_months, ncols=n_per_month, figure=fig)
    last_im = None

    for row, m in enumerate(months):
        for col, idx in enumerate(random_indices[m]):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            label = (
                month_names[m - 1]
                if month_names and 1 <= m <= len(month_names)
                else f"Month {m}"
            )
            last_im = plot_sample_on_ax(
                ax, X[idx, 0, :, :], extent,
                title=f"{label} | idx {idx}",
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            if col > 0:
                ax.set_ylabel("")

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Temperature Anomaly (°C)", fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Seasonal cycle diagnostics
# ---------------------------------------------------------------------------

def seasonal_cycle_before_after(
    X_raw: np.ndarray,
    X_ae_conv: np.ndarray,
    month_idx: np.ndarray,
    region_masks: dict | None = None,
    figtitle: str = "Seasonal cycle before/after anomaly",
    units: str = "°C",
    subplot_kwargs: dict | None = None,
    show_spaghetti: bool = True,
    hist_bins: int = 40,
    hist_range: tuple | None = None,
    inset_width: str = "30%",
    inset_height: str = "35%",
    inset_loc: str = "upper right",
) -> plt.Figure:
    """
    Compare raw vs anomaly seasonal cycle across regions with inset histograms.

    Parameters
    ----------
    X_raw, X_ae_conv : np.ndarray  shape ``(T, 1, n_lat, n_lon)``
    month_idx : np.ndarray  length ``T``, values 0–11
    region_masks : dict[str, np.ndarray(bool)] or None
    figtitle, units, subplot_kwargs, show_spaghetti : as named
    hist_bins, hist_range, inset_width, inset_height, inset_loc : histogram inset config

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_raw = _to_conv(X_raw)
    X_ae_conv = _to_conv(X_ae_conv)
    assert X_raw.shape == X_ae_conv.shape, "Shapes must match"

    Xr = X_raw[:, 0, :, :]
    Xa = X_ae_conv[:, 0, :, :]
    T, n_lat, n_lon = Xr.shape

    if region_masks is None:
        region_masks = {"Global": np.ones((n_lat, n_lon), dtype=bool)}
    region_masks = dict(region_masks)
    n_regions = len(region_masks)

    region_anom = {
        name: Xa[:, mask].mean(axis=1) for name, mask in region_masks.items()
    }
    if hist_range is None:
        all_vals = np.concatenate(list(region_anom.values()))
        lo, hi = np.percentile(all_vals[np.isfinite(all_vals)], [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        hist_range = (lo - pad, hi + pad)

    default_kw = {"nrows": 1, "ncols": n_regions,
                  "figsize": (5 * n_regions, 4), "squeeze": False}
    if subplot_kwargs:
        default_kw.update(subplot_kwargs)
    fig, axes = plt.subplots(**default_kw)
    axes = axes.flatten()
    months = np.arange(12)

    for ax_idx, (name, mask) in enumerate(region_masks.items()):
        ax = axes[ax_idx]
        Xr_region = Xr[:, mask].mean(axis=1)
        Xa_region = region_anom[name]
        raw_offset = Xr_region.mean()

        if show_spaghetti:
            for y in range(T // 12):
                slc = slice(12 * y, min(12 * (y + 1), T))
                ax.plot(month_idx[slc] + 1, Xr_region[slc] - raw_offset,
                        color="steelblue", alpha=0.05)
                ax.plot(month_idx[slc] + 1, Xa_region[slc],
                        color="red", alpha=0.05)

        clim_raw  = np.array([(Xr_region[month_idx == m] - raw_offset).mean() for m in months])
        clim_anom = np.array([Xa_region[month_idx == m].mean() for m in months])

        ax.plot(months + 1, clim_raw,  "-o", label="Raw (offset)", color="steelblue", lw=1.5)
        ax.plot(months + 1, clim_anom, "-o", label="Anomaly",      color="red",       lw=1.5)
        ax.axhline(0, color="k", lw=0.7)

        if ax_idx % default_kw["ncols"] == 0:
            ax.set_ylabel(f"Mean {units}")
        ax.set_xlabel("Month")
        ax.set_title(f"{name} seasonal cycle")
        ax.legend(fontsize=8, loc="upper left")

        ax_inset = inset_axes(ax, width=inset_width, height=inset_height,
                              loc=inset_loc, borderpad=0.8)
        ax_inset.hist(Xa_region, bins=hist_bins, range=hist_range,
                      density=True, color="gray", alpha=0.7, edgecolor="none")
        ax_inset.axvline(0, color="k", lw=0.5)
        ax_inset.set_title("PDF", fontsize=7)
        ax_inset.tick_params(axis="both", labelsize=6)
        ax_inset.set_yticklabels([])

    for ax in axes[n_regions:]:
        ax.set_visible(False)

    fig.suptitle(figtitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Core anomaly statistics
# ---------------------------------------------------------------------------

def summarize_anomaly_field_core(
    X_ae_conv: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    extent: list[float],
    month_idx: np.ndarray | None = None,
    units: str = "°C",
    title: str = "Anomaly statistics (core)",
    seasons: dict | None = None,
) -> tuple[dict, plt.Figure]:
    """
    Variance, skewness, and kurtosis diagnostic figure with optional seasonal breakdown.

    Parameters
    ----------
    X_ae_conv : np.ndarray  shape ``(T, 1, n_lat, n_lon)`` or ``(T, n_lat, n_lon)``
    lats, lons : np.ndarray
    extent : list[float]  ``[lon_min, lon_max, lat_min, lat_max]``
    month_idx : np.ndarray or None  length ``T``, values 0–11
    units, title, seasons : as named

    Returns
    -------
    stats : dict
    fig : matplotlib.figure.Figure
    """
    X_ae_conv = _to_conv(X_ae_conv)
    X_2d = X_ae_conv[:, 0, :, :]
    n_t, n_lat, n_lon = X_2d.shape

    var_map  = X_2d.var(axis=0)
    skew_map = skew(X_2d, axis=0)
    kurt_map = kurtosis(X_2d, axis=0, fisher=True)

    var_by_lat  = var_map.mean(axis=1)
    skew_by_lat = skew_map.mean(axis=1)
    kurt_by_lat = kurt_map.mean(axis=1)

    if month_idx is not None:
        if seasons is None:
            seasons = {"DJF": [11, 0, 1], "MAM": [2, 3, 4],
                       "JJA": [5, 6, 7],  "SON": [8, 9, 10]}
        var_seas, skew_seas, kurt_seas = {}, {}, {}
        for sname, mons in seasons.items():
            mask = np.isin(month_idx, mons)
            if mask.sum() < 2:
                var_seas[sname]  = np.full_like(var_map, np.nan)
                skew_seas[sname] = np.full_like(skew_map, np.nan)
                kurt_seas[sname] = np.full_like(kurt_map, np.nan)
            else:
                Xs = X_2d[mask]
                var_seas[sname]  = Xs.var(axis=0)
                skew_seas[sname] = skew(Xs, axis=0)
                kurt_seas[sname] = kurtosis(Xs, axis=0, fisher=True)
        n_cols = 3
        season_order = list(seasons.keys())
    else:
        var_seas = skew_seas = kurt_seas = {}
        season_order = []
        n_cols = 2

    fig = plt.figure(figsize=(18 if n_cols == 3 else 14, 10))
    width_ratios = [1.2, 1.4, 0.9] if n_cols == 3 else [1.2, 0.9]
    gs = GridSpec(3, n_cols, figure=fig, width_ratios=width_ratios,
                  height_ratios=[1.0, 1.0, 1.0], wspace=0.25, hspace=0.35)

    def _style(ax):
        ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", lw=0.5, zorder=10)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(lw=0.5)

    def _season_axes(row):
        gss = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row, 1],
                                      wspace=0.1, hspace=0.15)
        return [fig.add_subplot(gss[r, c], projection=ccrs.PlateCarree())
                for r in range(2) for c in range(2)]

    row_specs = [
        (var_map,  var_seas,  var_by_lat,  "YlOrRd", f"Variance ({units}²)",  "steelblue"),
        (skew_map, skew_seas, skew_by_lat, "RdBu_r", "Skewness",               "darkorange"),
        (kurt_map, kurt_seas, kurt_by_lat, "PuOr_r", "Excess kurtosis",        "purple"),
    ]

    for row_idx, (stat_map, seas_maps, by_lat, cmap, label, lat_color) in enumerate(row_specs):
        ax_map = fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree())
        if row_idx == 0:
            vmax_v = np.nanpercentile(stat_map, 99)
            im = ax_map.imshow(stat_map, origin="lower", aspect="auto",
                               extent=extent, cmap=cmap, vmin=0, vmax=vmax_v)
        else:
            sym = np.nanpercentile(np.abs(stat_map[np.isfinite(stat_map)]), 99)
            im = ax_map.imshow(stat_map, origin="lower", aspect="auto",
                               extent=extent, cmap=cmap, vmin=-sym, vmax=sym)
        _style(ax_map)
        plt.colorbar(im, ax=ax_map, label=label)

        if n_cols == 3 and season_order:
            s_axes = _season_axes(row_idx)
            for ax_s, sname in zip(s_axes, season_order):
                s_map = seas_maps[sname]
                if row_idx == 0:
                    ax_s.imshow(s_map, origin="lower", aspect="auto",
                                extent=extent, cmap=cmap, vmin=0, vmax=vmax_v)
                else:
                    ax_s.imshow(s_map, origin="lower", aspect="auto",
                                extent=extent, cmap=cmap, vmin=-sym, vmax=sym)
                _style(ax_s)
                ax_s.set_title(sname, fontsize=9)
                ax_s.set_xticklabels([])
                ax_s.set_yticklabels([])

        ax_lat = fig.add_subplot(gs[row_idx, n_cols - 1])
        ax_lat.plot(by_lat, lats, color=lat_color)
        ax_lat.axvline(0, color="k", lw=0.5)
        ax_lat.axhline(0, color="k", lw=0.5, ls="--")
        ax_lat.set_xlabel(f"Zonal mean {label.lower()}")
        ax_lat.set_ylabel("Latitude (°)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    stats = dict(
        var_map=var_map, skew_map=skew_map, kurt_map=kurt_map,
        var_seasonal=var_seas, skew_seasonal=skew_seas, kurt_seasonal=kurt_seas,
        var_by_lat=var_by_lat, skew_by_lat=skew_by_lat, kurt_by_lat=kurt_by_lat,
    )
    return stats, fig


def compare_field_statistics(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    extent: list[float],
    units: str = "°C",
    title: str = "Original vs reconstruction: anomaly statistics",
) -> tuple[dict, plt.Figure]:
    """
    Side-by-side variance/skewness/kurtosis comparison: original vs reconstructed field.

    Useful for evaluating how well an AE or PCA reconstruction preserves the
    higher-order statistics of the training data.

    Parameters
    ----------
    X_orig, X_recon : np.ndarray  both ``(T, 1, n_lat, n_lon)`` or ``(T, n_lat, n_lon)``
    lats, lons : np.ndarray
    extent : list[float]
    units, title : str

    Returns
    -------
    stats : dict
        Keys ``"orig"``, ``"recon"`` (each with ``var_map``, ``skew_map``,
        ``kurt_map``) and ``"preservation"`` (weighted MAE scalars).
    fig : matplotlib.figure.Figure
    """
    X_orig  = _to_conv(X_orig)
    X_recon = _to_conv(X_recon)
    assert X_orig.shape == X_recon.shape, "Shapes must match"

    def _maps(X4d):
        X2d = X4d[:, 0, :, :]
        return X2d.var(axis=0), skew(X2d, axis=0), kurtosis(X2d, axis=0, fisher=True)

    var_o, skew_o, kurt_o = _maps(X_orig)
    var_r, skew_r, kurt_r = _maps(X_recon)

    cos_w = np.cos(np.deg2rad(lats))
    cos_w = cos_w / cos_w.sum()

    def _wmae(a, b):
        return float((np.abs(a - b) * cos_w[:, np.newaxis]).sum() / len(lons))

    preservation = dict(
        var_mae  =_wmae(var_o,  var_r),
        skew_mae =_wmae(skew_o, skew_r),
        kurt_mae =_wmae(kurt_o, kurt_r),
    )

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    col_labels = ["Original", "Reconstructed", "Diff (recon − orig)"]

    for row_idx, (orig_m, recon_m, cmap, label) in enumerate([
        (var_o,  var_r,  "YlOrRd", f"Variance ({units}²)"),
        (skew_o, skew_r, "RdBu_r", "Skewness"),
        (kurt_o, kurt_r, "PuOr_r", "Excess kurtosis"),
    ]):
        diff_m = recon_m - orig_m
        if row_idx == 0:
            vmax = np.nanpercentile(
                np.concatenate([orig_m.ravel(), recon_m.ravel()]), 99
            )
            vmin = 0.0
        else:
            sym = np.nanpercentile(
                np.abs(np.concatenate([orig_m.ravel(), recon_m.ravel()])), 99
            )
            vmin, vmax = -sym, sym
        diff_vmax = np.nanpercentile(np.abs(diff_m), 99)

        for col_idx, (data, vlo, vhi, cm) in enumerate([
            (orig_m,  vmin, vmax,     cmap),
            (recon_m, vmin, vmax,     cmap),
            (diff_m, -diff_vmax, diff_vmax, "RdBu_r"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(data, origin="lower", aspect="auto",
                           extent=extent, cmap=cm, vmin=vlo, vmax=vhi)
            ax.add_feature(cfeature.LAND, facecolor="none",
                           edgecolor="black", lw=0.4, zorder=10)
            plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    stats = dict(
        orig  =dict(var_map=var_o, skew_map=skew_o, kurt_map=kurt_o),
        recon =dict(var_map=var_r, skew_map=skew_r, kurt_map=kurt_r),
        preservation=preservation,
    )
    return stats, fig


# ---------------------------------------------------------------------------
# QQ plots
# ---------------------------------------------------------------------------

def qqplot_anomaly_field(
    X_ae_conv: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    region_masks: dict | None = None,
    n_time_sample: int = 10_000,
    title: str = "QQ-plot of anomalies vs N(0,1)",
    subplot_kwargs: dict | None = None,
) -> plt.Figure:
    """
    QQ-plots of regional anomaly samples vs a standard normal.

    Parameters
    ----------
    X_ae_conv : np.ndarray  shape ``(T, 1, n_lat, n_lon)``
    lats, lons : np.ndarray
    region_masks : dict[str, np.ndarray(bool)] or None
    n_time_sample : int  max samples per region (for speed)
    title, subplot_kwargs : as named

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_ae_conv = _to_conv(X_ae_conv)
    X_2d = X_ae_conv[:, 0, :, :]
    _, n_lat, n_lon = X_2d.shape

    if region_masks is None:
        region_masks = {"Global": np.ones((n_lat, n_lon), dtype=bool)}
    n_regions = len(region_masks)

    default_kw = {"nrows": 1, "ncols": n_regions,
                  "figsize": (5 * n_regions, 4), "squeeze": False}
    if subplot_kwargs:
        default_kw.update(subplot_kwargs)

    fig, axes = plt.subplots(**default_kw)
    axes = axes.flatten()

    for ax, (name, mask) in zip(axes, region_masks.items()):
        data = X_2d[:, mask].reshape(-1)
        mu, sigma = np.nanmean(data), np.nanstd(data)
        z = (data - mu) / sigma
        if z.size > n_time_sample:
            idx = np.random.choice(z.size, size=n_time_sample, replace=False)
            z = z[idx]
        (osm, osr), (slope, intercept, r) = probplot(z, dist="norm")
        ax.plot(osm, osr, "o", ms=3, alpha=0.5, label=f"Data ({name})")
        ax.plot(osm, slope * osm + intercept, "r-", lw=1.5,
                label=f"Fit (R²≈{r**2:.2f})")
        ax.axline((0, 0), slope=1, color="k", lw=1, ls="--", label="1:1")
        ax.set_title(name)
        ax.set_xlabel("Theoretical quantiles (N(0,1))")
        ax.set_ylabel("Sample quantiles (std anomalies)")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# EOF analysis
# ---------------------------------------------------------------------------

def compute_eofs(
    X_ae_conv: np.ndarray,
    n_modes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute leading EOF patterns via PCA.

    Parameters
    ----------
    X_ae_conv : np.ndarray  shape ``(T, 1, n_lat, n_lon)``
    n_modes : int

    Returns
    -------
    eofs     : np.ndarray  ``(n_modes, n_lat, n_lon)``
    pcs      : np.ndarray  ``(T, n_modes)``
    var_frac : np.ndarray  ``(n_modes,)``
    """
    X_ae_conv = _to_conv(X_ae_conv)
    X_2d = X_ae_conv[:, 0, :, :]
    n_t, n_lat, n_lon = X_2d.shape
    X_flat = X_2d.reshape(n_t, -1)
    X_flat = X_flat - X_flat.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_modes)
    pcs = pca.fit_transform(X_flat)
    eofs = pca.components_.reshape(n_modes, n_lat, n_lon)
    return eofs, pcs, pca.explained_variance_ratio_


def plot_eof_maps(
    eofs: np.ndarray,
    var_frac: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    extent: list[float],
    title: str | None = None,
    subplot_kwargs: dict | None = None,
) -> plt.Figure:
    """
    Visualise leading EOF spatial patterns.

    Parameters
    ----------
    eofs : np.ndarray  ``(n_modes, n_lat, n_lon)``
    var_frac : np.ndarray  ``(n_modes,)``
    lats, lons : np.ndarray
    extent : list[float]
    title, subplot_kwargs : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_modes = eofs.shape[0]
    n_cols  = min(3, n_modes)
    n_rows  = int(np.ceil(n_modes / n_cols))

    default_kw = {
        "nrows": n_rows, "ncols": n_cols,
        "figsize": (5 * n_cols, 3.5 * n_rows),
        "subplot_kw": {"projection": ccrs.PlateCarree()},
        "squeeze": False,
    }
    if subplot_kwargs:
        default_kw.update(subplot_kwargs)

    fig, axes = plt.subplots(**default_kw)
    for i, ax in enumerate(axes.ravel()):
        if i >= n_modes:
            ax.set_visible(False)
            continue
        pattern = eofs[i]
        vmax = np.max(np.abs(pattern))
        im = ax.imshow(pattern, origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.add_feature(cfeature.LAND, facecolor="none",
                       edgecolor="black", lw=0.5, zorder=10)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"EOF {i+1} ({var_frac[i]*100:.1f}% var)")

    if title is None:
        title = (f"Leading {n_modes} EOF patterns "
                 f"({var_frac.sum()*100:.1f}% total variance)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Temporal structure
# ---------------------------------------------------------------------------

def compute_temporal_stats_from_flat(
    X_ae: np.ndarray,
    lat_bands: dict,
    city_flat: dict,
) -> tuple[dict, dict, dict]:
    """
    Temporal statistics and regional time series from flattened ``X_ae``.

    Parameters
    ----------
    X_ae : np.ndarray  shape ``(T, n_lat * n_lon)``
    lat_bands : dict
        label → dict with ``flat_start``, ``flat_end``, ``color``,
        ``ls``, ``actual_lat``.
    city_flat : dict
        city_name → dict with ``flat_idx``.

    Returns
    -------
    ts_stats : dict  (mean, std, min, max, median — all length T)
    band_ts  : dict  label → 1-D array
    city_ts  : dict  city_name → 1-D array
    """
    ts_stats = dict(
        mean   =X_ae.mean(axis=1),
        std    =X_ae.std(axis=1),
        min    =X_ae.min(axis=1),
        max    =X_ae.max(axis=1),
        median =np.median(X_ae, axis=1),
    )
    band_ts = {
        label: X_ae[:, b["flat_start"]: b["flat_end"] + 1].mean(axis=1)
        for label, b in lat_bands.items()
    }
    city_ts = {
        cname: X_ae[:, c["flat_idx"]]
        for cname, c in city_flat.items()
    }
    return ts_stats, band_ts, city_ts


def plot_temporal_structure_flat(
    t_num: np.ndarray,
    ts_stats: dict,
    band_ts: dict,
    city_ts: dict,
    lat_bands: dict,
    city_flat: dict,
    units: str = "°C",
    zoom_len: int = 120,
    title: str = "Temporal statistics & regional anomaly time series",
) -> plt.Figure:
    """
    3×3 temporal structure figure: global stats, lat-band means, city series.

    Parameters
    ----------
    t_num : np.ndarray  numeric time axis (e.g. fractional years)
    ts_stats : dict  from :func:`compute_temporal_stats_from_flat`
    band_ts, city_ts : dict
    lat_bands, city_flat : dict
    units, zoom_len, title : as named

    Returns
    -------
    matplotlib.figure.Figure
    """
    ts_mean   = ts_stats["mean"]
    ts_std    = ts_stats["std"]
    ts_min    = ts_stats["min"]
    ts_max    = ts_stats["max"]
    ts_median = ts_stats["median"]
    T = len(t_num)

    t_s1, t_e1 = t_num[0], t_num[min(zoom_len, T - 1)]
    t_s2, t_e2 = t_num[max(0, T - zoom_len)], t_num[-1]

    fig = plt.figure(figsize=(22, 12), facecolor="#ffffff")
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.15,
                  height_ratios=[2.0, 1.5, 1.5], width_ratios=[1.2, 1.2, 3.0])

    def _fill_global(ax, xlim=None):
        ax.fill_between(t_num, ts_mean - ts_std, ts_mean + ts_std,
                        color="orange", alpha=0.20, zorder=1)
        ax.plot(t_num, ts_min,    color="steelblue",    lw=0.8, ls=":",  alpha=0.75, zorder=2)
        ax.plot(t_num, ts_max,    color="mediumpurple", lw=0.8, ls=":",  alpha=0.75, zorder=2)
        ax.plot(t_num, ts_median, color="green",        lw=1.0, ls="-.", alpha=0.85, zorder=3)
        ax.plot(t_num, ts_mean,   color="red",          lw=1.2, ls="--", alpha=0.90, zorder=4)
        if xlim:
            ax.set_xlim(*xlim)

    ax1z1 = fig.add_subplot(gs[0, 0])
    _fill_global(ax1z1, (t_s1, t_e1))
    ax1z1.set_title("First 120 Months", fontsize=11, fontweight="semibold", loc="left")
    ax1z1.set_ylabel(f"Temp Anomaly ({units})", fontsize=10)

    ax1z2 = fig.add_subplot(gs[0, 1], sharey=ax1z1)
    _fill_global(ax1z2, (t_s2, t_e2))
    ax1z2.set_title("Last 120 Months", fontsize=11, fontweight="semibold", loc="left")

    ax1 = fig.add_subplot(gs[0, 2], sharey=ax1z1)
    ax1.fill_between(t_num, ts_mean - ts_std, ts_mean + ts_std,
                     color="orange", alpha=0.20, zorder=1, label="mean ± std")
    ax1.plot(t_num, ts_min,    color="steelblue",    lw=0.8, ls=":",  alpha=0.75, label="Min",    zorder=2)
    ax1.plot(t_num, ts_max,    color="mediumpurple", lw=0.8, ls=":",  alpha=0.75, label="Max",    zorder=2)
    ax1.plot(t_num, ts_median, color="green",        lw=1.0, ls="-.", alpha=0.85, label="Median", zorder=3)
    ax1.plot(t_num, ts_mean,   color="red",          lw=1.2, ls="--", alpha=0.90, label="Mean",   zorder=4)
    ax1.set_title("Global spatial statistics", fontsize=12, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.75, ncol=5)

    def _plot_bands(ax):
        for label, ts in band_ts.items():
            b = lat_bands[label]
            ax.plot(t_num, ts, color=b["color"], lw=1.1, ls=b["ls"], alpha=0.85)

    ax2z1 = fig.add_subplot(gs[1, 0], sharex=ax1z1)
    _plot_bands(ax2z1)
    ax2z1.set_ylabel(f"Mean Anomaly ({units})", fontsize=10)
    ax2z1.set_title("Lat-band mean anomaly", fontsize=11, fontweight="semibold", loc="left")

    ax2z2 = fig.add_subplot(gs[1, 1], sharex=ax1z2, sharey=ax2z1)
    _plot_bands(ax2z2)

    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax2z1)
    for label, ts in band_ts.items():
        b = lat_bands[label]
        ax2.plot(t_num, ts, color=b["color"], lw=1.1, ls=b["ls"], alpha=0.85,
                 label=f"{label} ({b['actual_lat']:+.2f}°)")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.75, ncol=3)

    def _plot_cities(ax):
        for cname, ts in city_ts.items():
            ax.plot(t_num, ts, lw=1.0, alpha=0.8)

    ax3z1 = fig.add_subplot(gs[2, 0], sharex=ax1z1)
    _plot_cities(ax3z1)
    ax3z1.set_xlabel("Year", fontsize=11)
    ax3z1.set_ylabel(f"Anomaly ({units})", fontsize=10)
    ax3z1.set_title("City gridpoint anomaly", fontsize=11, fontweight="semibold", loc="left")

    ax3z2 = fig.add_subplot(gs[2, 1], sharex=ax1z2, sharey=ax3z1)
    _plot_cities(ax3z2)
    ax3z2.set_xlabel("Year", fontsize=11)

    ax3 = fig.add_subplot(gs[2, 2], sharex=ax1, sharey=ax3z1)
    for cname, ts in city_ts.items():
        c = city_flat[cname]
        ax3.plot(t_num, ts, lw=1.0, alpha=0.8,
                 label=f"{cname} ({c['actual_lat']:+.1f}°, {c['actual_lon']:+.1f}°)")
    ax3.set_xlabel("Year", fontsize=11)
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.75, ncol=2)

    all_ax = [ax1z1, ax1z2, ax1, ax2z1, ax2z2, ax2, ax3z1, ax3z2, ax3]
    for ax in all_ax:
        ax.axhline(0, color="#888888", lw=0.6, alpha=0.5)
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors="#333333", labelsize=9)

    for ax in [ax1z2, ax1, ax2z2, ax2, ax3z2, ax3]:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.yaxis.set_tick_params(left=False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
