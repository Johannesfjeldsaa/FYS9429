# Alternative formulations of monthly output
Sandstad et al. (2025) introduced METEORv1.0.1, a pulse response model with capabilities of generating spatially resolved annual outputs, further referred to as METEOR-CORE. In prep work from Sanderson et al. (2026) extended on METEOR-CORE adding a noise module which uses a PCA-VARX stacked modelling to give monthly output. In this directory we will explore alternative formulation of the noise modell still focusing on monthly resolution. We will do this using variants of autoencoders and variants of the training targets.

## README contents:

1. Data flow/pipeline description of METEORv1.6 (in prep.) which describes how METEOR-CORE is extended upon using PCA-VARX and how ensemble generation is performed.
2. Descriptions of alternative formulations
* Testing the normality assumption of residual fields with student-T test
* Attempting to model the non-linear part of residuals from the PCA-VARX only using AE.
* Attempting to replace the full PCA model using non-linear autoencoders that are physics informed.
* Attempting to replace the Full PCA-VARX model by condintional convolutional variational autoencoder.

# Pipeline description METEORv1.6

# METEORv1.6 Standard Architecture: Data Flow and Pipeline

*(from Sanderson et al. (2026, in prep.))*

## PHASE 1: METEOR-CORE Forced Response Generation

```bash
┌─────────────────────────────────────────────────────────────────────────────┐
│ METEOR-CORE Pattern Scaling                                                 │
│  • Multi-timescale impulse response formulation                             │
│  • Annual mean predictions: X̄_METEOR(t_a, x, y)                             │
│    where t_a = year index ⌊(t−1)/12⌋+1                                      │
│  • Forced response + long-term trend (excludes internal variability)        │
│  • Replicated 12× for monthly timesteps                                     |
│                                                                             │
│ Output: (n_years, n_lat, n_lon) annual fields → broadcast to monthly        │
└─────────────────────────────────────────────────────────────────────────────┘
```
## PHASE 2: Seasonal-Anomaly Decomposition (METEOR-NOISE Backbone)

```bash
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT: CMIP6 Training Data (historical + SSP2-4.5)                          │
│        Dimensions: (n_months, n_lat, n_lon)                                 │
│        ~250 years monthly data, single ensemble member                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE A: Global Temperature Extraction                                      │
│  • Area-weighted mean: T_glob(t) = Σ[w_i·X_i]/Σ[w_i]                        │
│    with w_i = cos(lat_i)                                                    │
│  • 5-year (60-month) smoothing to remove high-frequency noise               │
│    while preserving multi-decadal trend                                     │
│                                                                             │
│ Output: T_glob_smooth(t) — smoothed global temperature trajectory           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE B: Temperature-Dependent Harmonic Seasonal Model                      │
│  • 9-feature construction per timestep:                                     │
│    - Direct term: T_glob(t)                                                 │
│    - Harmonic terms: cos(2πt/12), sin(2πt/12), cos(4πt/12), sin(4πt/12)     │
│    - Modulated terms: T_glob(t)·cos(2πt/12), T_glob(t)·sin(2πt/12), etc.    │
│                                                                             │
│  • Linear regression per gridpoint:                                         │
│    X(t,x,y) = β₀(x,y) + Σⱼ₌₁⁹ βⱼ(x,y)·X_harm,j(t) + ε(t,x,y)                │
│                                                                             │
│  • Seasonal variance explained: ~91% temperature, ~32% precipitation        │
│                                                                             │
│ Output: Ŷ(t,lat,lon) — predicted seasonal cycle; fitted coefficients β      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE C: Anomaly Extraction                                                 │
│  • Residuals: A(t,x,y) = X(t,x,y) − Ŷ(t,x,y) − baseline                     │
│  • Represents internal variability (forced response removed)                │
│  • piControl baseline subtraction for consistency                           │
│                                                                             │
│ Output: A(t,lat,lon) — monthly anomaly fields                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## PHASE 3: Standard PCA Decomposition (METEORv1.6)

```bash
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE D: Principal Component Analysis (Standard 40 Modes)                     │
│  • Flatten anomalies: A_flat(t,s) where s = lat×lon gridpoints                │
│    Shape: (n_months, ~10,000+ gridpoints) — e.g., (3012, 13824)               │
│                                                                               │
│  • Decomposition: A_flat = Z·Eᵀ + μ                                           │
│    - Z: PC time series (n_months, n_modes) — default n_modes = 40             │
│    - E: EOF spatial patterns (n_modes, n_gridpoints)                          │
│    - μ: spatial mean (~0 after seasonal removal)                              │
│                                                                               │
│  • Variance explained by 40 PCs                                               │
│    - Temperature: 75-80% of anomaly variance (98% total with seasonal)        │
│    - Precipitation: 29-45% of anomaly variance (51-64% total)                 │
│                                                                               │
│  • Key property: Linear projection enables efficient regional calculations    │
│    without full spatial reconstruction                                        │
│                                                                               │
│ Output: {Z(t), E(x,y), μ, explained_variance}                                 │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
```
PCA representation of the anomalies has several advantages. First, it achieves massive dimensionality reduction: instead of modeling 10,000+ grid points, we model 40 timeseries. Second, the spatial orthogonality simplifies subsequent statistical modeling.

## PHASE 4: VARX Temporal Modeling

```bash
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE E: VARX(2) Model Fitting                                                │
│  • Model equation                                                             │
│    Z(t) = c + A₁·Z(t−1) + A₂·Z(t−2) + B·X_exog(t) + ε(t)                      │
│    where ε(t) ~ N(0, Σ)                                                       │
│                                                                               │
│  • Exogenous variables X_exog(t)                                              │
│    - use_exog='temp_only': [T_glob(t)] — temperature modulation               │
│    - use_exog='all': [T_glob(t), cos(2πt/12), sin(2πt/12)] — with harmonics   │
│    - use_exog='none': pure VAR (default for precipitation stability)          │
│                                                                               │
│  • Coefficient dimensions:                                                    │
│    - c: (n_modes,) — intercept                                                │
│    - A₁, A₂: (n_modes, n_modes) — lag-1 and lag-2 autoregressive matrices     │
│    - B: (n_modes, n_exog) — exogenous coefficients                            │
│    - Σ: (n_modes, n_modes) — residual covariance (captures teleconnections)   │
│                                                                               │
│  • Stationarity assumption                                                    │
│    Lag coefficients (A₁, A₂) assumed constant in time                         │
│                                                                               │
│ Output: {c, A₁, A₂, B, Σ} — VARX parameters                                   │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
```
Each PC's prediction then depends on lagged values of all 40 PCs and the global trend, not just itself. This allows for richer teleconnections: if tropical Pacific warming represented in one PC typically precedes North American circulation changes (represented in another PC) by one month, the model learns a non-zero coefficient connecting these PCs at lag 1. The model thus has thousands of coefficients—40 PCs times 2 lags times 40 PCs plus exogenous terms—estimated efficiently using standard least-squares regression.



---

## PHASE 5: Ensemble Generation

```bash
┌───────────────────────────────────────────────────────────────────────────────┐
│ INPUT: Target Temperature Trajectory T_glob_future(t)                         │
│        From METEOR-CORE or external GMT scenario                              │
│        Shape: (n_time,) — e.g., 1200 months (100 years)                       │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE F: Synthetic PC Generation (VARX Simulation)                            │
│  • Initialize: Z(0) = Z(1) = 0 (neutral initial conditions)                   │
│    — Ensures ensemble members start from climatology and diverge naturally    │
│                                                                               │
│  • Pre-generate random shocks: ε(t) ~ MVN(0, Σ) for all t                     │
│    — Single batched draw for computational efficiency                         │
│    — Vectorized operations avoid explicit loops over ensemble members         │
│                                                                               │
│  • Autoregressive loop (vectorized):                                          │
│    For t = 2, ..., n_time-1:                                                  │
│      Z(t) = c + A₁·Z(t−1) + A₂·Z(t−2) + B·X_exog(t) + ε(t)                    │
│                                                                               │
│  • Independent realizations: Different random seeds for each ensemble member  │
│    — Members share same VARX parameters but diverge due to ε(t) sampling      │
│                                                                               │
│ Output: {Z(t)} — simulated PC time series for N realizations                  │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE G: Spatial Reconstruction and Variable-Specific Processing              │
│                                                                               │
│  Step 1: Anomaly reconstruction via PCA                                       │
│    A(t,x,y) = Σₖ₌₁⁴⁰ Zₖ(t)·Eₖ(x,y)                                            │
│    — Matrix multiplication: (n_time, 40) × (40, n_gridpoints)                 │
│                                                                               │
│  Step 2: Seasonal cycle computation                                           │
│    Ŷ(t,x,y) = Σⱼ₌₂⁹ βⱼ(x,y)·X_harm,j(t)  [noise-only: excludes direct T term] │
│    — Uses fitted β from Stage B with new trajectory's X_harm features         │
│                                                                               │
│  Step 3: Combine with METEOR-CORE forced response                             │
│    X_ensemble(t,x,y) = X̄_METEOR(t_a,x,y) + Ŷ(t,x,y) + A(t,x,y)                │
│    where t_a = ⌊(t−1)/12⌋+1 converts month to year index                      │
│                                                                               │
│  Step 4: Variable-specific transforms                                         │
│    — Temperature (tas): Direct output (Gaussian anomalies)                    │
│    — Precipitation (pr): Gamma transform                                      │
│      P_transformed = F_Γ⁻¹(Φ((P_noise−μ)/σ); k, θ)                            │
│      where F_Γ⁻¹ is inverse gamma CDF, Φ is standard normal CDF               │
│      ensures non-negativity and preserves right-skewed distribution           │
│                                                                               │
│ Output: X_ensemble(t,lat,lon) — full spatial climate fields                   │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE H: Ensemble Replication and Caching                                     │
│  • Repeat Stages F-G for N realizations (default N=100)                       │
│    — 100 members × 350 years takes ~90 seconds for regional means             │
│    — ~10 seconds for single point extraction                                  │
│                                                                               │
│  • Memory optimization: Store only PCs (Z), reconstruct spatial on demand     │
│    — Reduces memory by 2-3 orders of magnitude                                │
│                                                                               │
│  • Hybrid storage approach:                                                   │
│    — Cache PCs for all members                                                │
│    — Reconstruct full fields only for regions of interest when needed         │
│                                                                               │
│ Output: Ensemble of N realizations (stored as PC trajectories + metadata)     │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Model Serialization (Standard METEORv1.6)

```bash
┌───────────────────────────────────────────────────────────────────────────────┐
│ TRAINED MODEL COMPONENTS (pickled/serialized)                                 │
│                                                                               │
│ meteor_model.pkl contains:                                                    │
│ {                                                                             │
│   'version': '1.6',                                                           │
│   'seasonal_coeffs': β (n_lat, n_lon, 9) — from Stage B                       │
│   'pca_components': {                                                         │
│     'EOF': E (40, n_lat, n_lon),                                              │
│     'mean': μ (n_lat, n_lon),                                                 │
│     'explained_variance': σ² (40,),                                           │
│     'n_modes': 40                                                             │
│   },                                                                          │
│   'varx_params': {                                                            │
│     'intercept': c (40,),                                                     │
│     'A1': (40, 40), 'A2': (40, 40), — lag coefficient matrices                │
│     'B': (40, n_exog), — exogenous coefficients                               │
│     'Sigma': (40, 40), — residual covariance                                  │
│     'lag_order': 2,                                                           │
│     'use_exog': 'temp_only' | 'all' | 'none'                                  │
│   },                                                                          │
│   'precomputed_regional_projections': {                                       │
│     'global': Ē_global (40,),                                                 │
│     'AR6_regions': {name: Ē_name (40,) for name in AR6_list},                 │
│     'point_locations': {(lat,lon): Ē_point (40,)}                             │
│   },                                                                          │
│   'variable_transforms': {                                                    │
│     'tas': {'type': 'identity', 'baseline': piControl_mean},                  │
│     'pr': {'type': 'gamma', 'params': {'k': shape, 'theta': scale}}           │
│   },                                                                          │
│   'training_metadata': {                                                      │
│     'source_model': 'CESM2-WACCM',                                            │
│     'training_scenario': 'ssp245',                                            │
│     'n_months': 3012,                                                         │
│     'grid_resolution': '96x144',                                              │
│     'variance_explained': {                                                   │
│       'seasonal': 0.91,                                                       │
│       'pca': 0.76,                                                            │
│       'total': 0.98                                                           │
│     },                                                                        │
│     'timestamp': '2024-01-15T10:30:00'                                        │
│   }                                                                           │
│ }                                                                             │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

# Alternative formulations of METEOR-Noise

There is a large difference in how much the variance METEOR is able to model between temperature and preciptation fields. Therefor our first order attempt on improving the noise modelling we differantiate:

* Temperature has approximately 98% of the total variance explained, it is however underdispersive in it's predictions. We therefor try to improve by challanging the assumption that the principal components are normal distributions; see [The normality assumption of principal components for temperature](#the-normality-assumption-of-principal-components-for-temperature) for more on this.
* Precipitation has approximately 51-64% of the total variance explained, and so there are mutliple areas to improve on here. The first attempt for improving this is therefor to try to introduce some non-linarity into the modelling. For this we introduce a hybrid decoposition of residual from seasonal harmonics model; see [non-linear AE-PCA-VARX for precipitation](#non-linear-ae-pca-varx-for-precipitation) for more on this.

For the two latter approaches; [non-linear AE-VARX](#non-linear-ae-varx) and [Conditional convolutional VAE](#conditional-convolutional-vae) we attempt different modelling approaches and apply them to both temperature and precipitation. For all approaches here we keep the seasonal harmonics part of METEOR-Noise.

## The normality assumption of principal components for temperature
**In a nutshell idea:** Keep the same PCA-VARX dynamics, but make the stochastic output more dispersive by replacing Gaussian PC marginals with heavier-tailed Student-t marginals through post-hoc scaling.

What is implemented:

1. Fit the usual seasonal model + PCA + VARX.
2. Simulate PCs with the same Gaussian VAR innovations as in the standard model.
3. In Student-t mode, scale each simulated PC at each timestep by a chi-squared factor:

$$
z_{t,j}^{(t)} = \sqrt{\frac{\nu_j}{V_{t,j}}}\, z_{t,j}^{(\mathrm{normal})},
\qquad V_{t,j} \sim \chi^2(\nu_j)
$$

Why this can help underdispersion:

- Underdispersion means extremes are too rare and ensemble spread is too narrow.
- Student-t scaling increases tail probability relative to Gaussian marginals, so rare but plausible deviations occur more often.
- Because scaling is applied after the VAR recursion, we keep the learned temporal structure while widening tails in the generated PCs.

Why Student-t is a better fit for PCs here (based on diagnostics):

- Mean excess kurtosis across PCs is positive (0.423), indicating heavier-than-Gaussian tails overall.
- The heaviest PC has excess kurtosis 1.733 (PC 7), with fitted df 4.7, which is strongly non-Gaussian.
- Heterogeneity is clear: median fitted df is 20.9, but max df is 6687639496.0 (PC 8, nearly Gaussian).
- 12/40 PCs have fitted df < 15, so a substantial subset needs heavier tails than a normal model provides.

Configuration used in code:

- noise_pc_distribution = normal: no post-hoc scaling.
- noise_pc_distribution = t: post-hoc scaling is applied.
- t_df = None: uses \(\nu_j = 10\) for all PCs.
- t_df = float: uses that \(\nu\) for all PCs.
- t_df = mle: estimates per-PC \(\nu_j\) from VARX residuals via univariate MLE.



## non-linear AE-PCA-VARX for precipitation
**In a nutshell idea:** We still keep the PCA-VARX model but instead of assuming linear relation to be sufficent we ...

## non-linear AE-VARX

**In a nutshell idea:** We still keep the VARX model but replace the PCA part of METEOR-Noise with a non-linear latent space.


## Conditional convolutional VAE

**In a nutshell idea:** We only keep spherical harmonics part to remove part of the seasonal cycle and model the whole spatial field using conditional convolutional variational AE's.

Here we will also explore the introduction of aditional predictors. This implies that METEOR-CORE needs to provide more variables, we will explore this in `modelling_new_variables_with_METEOR.ipynb`.
