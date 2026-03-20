# Project: Mixed-Frequency Dynamic Factor Model (MFDFM)

## Goal
Implement the large-scale mixed-frequency dynamic factor model described in:
> Galli, A. (2017). *Which indicators matter? Analyzing the Swiss business cycle using a large-scale mixed-frequency dynamic factor model.* SNB Working Paper 8/2017.

File: `MFDFM_Swiss_Business_Cycle.pdf`

The model produces a **monthly business cycle index** for Switzerland by combining monthly and quarterly indicators via a state-space DFM estimated with the Kalman smoother.

---

## Environment

- **Container**: `node:20` devcontainer, workspace mounted at `/workspace`
- **Python**: 3.11.2 — available as `python3` or `python3.11`
- **Pre-installed Python packages**: `pypdf` (system-wide via pip3)
- **Venv**: use `python3 -m venv .venv` then `source .venv/bin/activate`
- **Shell**: zsh
- **Version control**: Git repo at `/workspace`, hosted on GitHub
- **Python packages**: Do NOT use `.venv` — the container has no internet access. Instead, add packages to both `requirements.txt` (for local use) and `.devcontainer/Dockerfile` (under the `pip3 install` block), then rebuild the container (`Dev Containers: Rebuild Container` in VS Code)

---

## Model Summary (Galli 2017)

### Data
- 620 variables total: n^M = 362 monthly, n^Q = 285 quarterly (some overlap — GDP is first element, with 27 quarterly GDP components included in n^Q)
- 17 indicator categories: GDP, labor market, consumption, investment, foreign trade, foreign activity, financial markets, prices, construction, retail trade, wholesale trade, accommodation, manufacturing, project engineering, banking, insurance, other services, other indicators
- ~80 foreign indicators (euro area, Germany, US, Japan, emerging Asia, CESifo world survey)
- 4 indicator types: (I) hard data (36%), (II) soft data / surveys (43%), (III) foreign indicators, (IV) financial indicators
- All data used in calendar-, seasonally-, and outlier-adjusted terms
- Data not available seasonally adjusted are adjusted using **X-13ARIMA-SEATS**
- Sample starts 1975; many Swiss series start only from ~1990 (ragged start)
- Ragged edge: indicators have varying publication lags (0 to 2–3 months)

### Data Transformations
- Non-stationary monthly indicators enter the model as **3-month differences** (not month-on-month)
- Non-stationary quarterly indicators enter as **quarter-on-quarter differences or growth rates**
- This choice of 3-month changes for monthly flow variables is important — it means the same time aggregation rule G(L) = 1/3 + 1/3 L + 1/3 L² applies to both quarterly stock AND flow variables

### Model Structure

**Data vectors** (eq. 1):
```
y_t  := [GDP_t; x^Q_t; x^M_t]     (observed, mixed-frequency)
y*_t := [gdp*_t; x^{Q*}_t; x^M_t] (latent monthly equivalent — all values exist)
```
- `y_t`: quarterly parts observed only every 3rd month (t=3,6,9,...), set to **zero** otherwise
- `y*_t`: fully latent monthly vector — quarterly values `gdp*_t` and `x^{Q*}_t` are unobserved between quarter-end months

**Observation equation** (monthly latent, eq. 2):
```
y*_t = Λ f_t + u_t
```
- `Λ`: n × r factor loadings matrix
- `f_t`: r × 1 vector of unobserved common factors
- `u_t`: n × 1 idiosyncratic errors, i.i.d. with `u_t ~ N(0, Σ_ww)` where `Σ_ww` is diagonal
- No autocorrelation modeled in idiosyncratic errors (Θ = 0 in eq. 3) — tested and found to worsen performance

**Factor dynamics** (eq. 4 — VAR(p)):
```
f_t = Φ_1 f_{t-1} + ... + Φ_p f_{t-p} + v_t,  v_t ~ N(0, Σ_vv)
```
where each `Φ_j` is r × r and `Σ_vv` is r × r.

### Time Aggregation (Section 2.2)

The link between quarterly observed values and latent monthly values uses the aggregation rule from **Angelini, Camba-Mendez, Giannone, Reichlin, and Runstler (2011)** — NOT the Mariano-Murasawa (2003) alternative (tested, found inferior for Switzerland).

**Aggregation polynomial** (eq. 5): for t = 3,6,9,...:
```
x^Q_{i,t} = G(L) x^{Q*}_{i,t} = (1/3 + 1/3 L + 1/3 L²) x^{Q*}_{i,t}
```
Quarterly level = simple average of 3 monthly latent levels within the quarter.

**Observation-to-latent mapping** (eqs. 6-7):

For quarter-end months (t = 3,6,9,...):
```
y_t = (G_0 + G_1 L + G_2 L²) y*_t

where G_0 = [1/3 I_{n^Q}  O    ]    G_1 = [1/3 I_{n^Q}  O]    G_2 = [1/3 I_{n^Q}  O]
             [O           I_{n^M}]          [O            O]          [O            O]
```

For non-quarter months (t ≠ 3,6,9,...):
```
y_t = G y*_t

where G = [O_{n^Q}  O    ]
          [O        I_{n^M}]
```
This zeros out quarterly rows in non-quarter months.

**Combined observation equation** (eq. 8-9):
```
y_t = G_t(L) y*_t = G_t(L)(Λ f_t + u_t) = G_t(L) Λ f_t + G_t(L) u_t

where G_t(L) = { G_0 + G_1 L + G_2 L²    for t = 3,6,9,...
               { G                          otherwise
```

### Hyperparameters (chosen by BIC, Section 2.3)
- **r = 4** factors — selected via BIC on GDP measurement equation (eq. 10):
  `BIC(r) = ln(V[G_t(L) û_{t,gdp}]) + ln(T)/T * r`
- **p = 1** lag in factor VAR — selected via BIC (eq. 11):
  `BIC(p) = ln|Σ̂_vv(p)| + lnT/T * p * r²`
- **No autocorrelation** in idiosyncratic errors (u_t is i.i.d.)

### State-Space Form (Section 2.4, eqs. 12-19)

**State vector**: `ξ_t = [f_t; f_{t-1}; f_{t-2}]` (dimension 3r × 1)

**Measurement equation** (eq. 12): `y_t = H_t ξ_t + ε_t`

```
H_t = { [G_0 Λ   G_1 Λ   G_2 Λ]        for t = 3,6,9,...    (n × 3r)
       { [G Λ     O_{n×2r}      ]        otherwise
```

**Measurement noise** (eq. 14): `ε_t ~ N(0_n, R)`

```
R = { [3/9 Σ^Q_ww    O_{n^Q × n^M}]     for t = 3,6,9,...
    { [O_{n^M × n^Q}  Σ^M_ww      ]
    {
    { [c I_{n^Q}       O_{n^Q × n^M}]    otherwise  (c = small constant for invertibility)
    { [O_{n^M × n^Q}   Σ^M_ww      ]

Note: 3/9 Σ^Q_ww = 1/3 Σ^Q_ww (since V[G(L)u_t] = 3/9 V(u_t) for quarterly vars)
```

**Transition equation** (eq. 15): `ξ_t = F ξ_{t-1} + e_t`

```
F = [Φ_1 ... Φ_p   O_{r×(3-p)r}]     (3r × 3r, with p=1:  [Φ  O  O])
    [I_{2r}          O_{2r×r}    ]                            [I  O  O]
                                                              [O  I  O]
```

**Transition noise** (eq. 16): `e_t ~ N(0_{3r}, Q)`

```
Q = [Σ_vv      O_{r×2r} ]
    [O_{2r×r}   O_{2r×2r}]
```

**Full system** (eqs. 17-19):
```
y_t  = H_t ξ_t + ε_t        (measurement)
ξ_t  = F ξ_{t-1} + e_t      (transition)

[e_t ]     ([Q  0])
[ε_t ] ~ N ([0  R])           e_t and ε_t uncorrelated
```

### Estimation (2-step procedure, Section 2.4)

**Step 1 — Parameter estimation via PCA + OLS:**

1. **PCA** on balanced monthly-only subsample (minimum 15 years) → first r principal components `f^{PC}_t`
2. **OLS** to estimate `Λ`:
   - Monthly vars: `x^M_{i,t} = Λ_i f^{PC}_t + u_{i,t}`
   - Quarterly vars: `x^Q_{i,t} = Λ_i G(L) f^{PC}_t + ũ_{i,t}` where `ũ_{i,t} = G(L) u_{i,t}`
3. **Estimate `Σ_ww`** from OLS residuals:
   - Monthly: `σ²_{ww,i} = V(û_{i,t})`
   - Quarterly: `σ²_{ww,i} = (9/3) V(ũ_{i,t})` — because `V[ũ_t] = V[G(L)u_t] = (1/9+1/9+1/9) V(u_t) = 3/9 V(u_t)`, so `V(u_t) = 9/3 V(ũ_t)`
4. **Estimate `Φ` and `Σ_vv`** via VAR(p) on first r principal components

**Step 2 — Kalman smoother:**

Run the Kalman smoother on the full mixed-frequency dataset using parameters from Step 1. Missing observations (non-quarter months for quarterly vars, ragged edge, ragged start) are handled by setting `y_{i,t} = 0` and adjusting `H_t` and `R` accordingly (see Durbin & Koopman, 2012).

### Business Cycle Index
- The BCI = **monthly fitted values for quarterly GDP** from the Kalman smoother
- Specifically: `BCI_t = Λ_gdp f_{t|T}` where `Λ_gdp` is the first row of `Λ` (GDP loadings) and `f_{t|T}` are the smoothed factors
- Correlation with GDP: 0.81 (1990-2016), 0.89 (2000-2016)

### BCI Accuracy (eq. 29)
```
V(BCI_t) = Λ_gdp P^f_{t|T} Λ'_gdp
```
where `P^f_{t|T}` is the first 4×4 block of the smoothed state covariance matrix (covariance of contemporaneous factors).
- Accuracy reaches ~55% at end of target month, ~90% at +30 days, ~99% at +60 days

### Indicator Weight Decomposition (Section 4.2, eqs. 20-22)
Smoothed factors can be decomposed as weighted sum of all observations:
```
f_{t|T} = Σ_{k=0}^{T} w_k(t,T) y_k
```
Weights computed via Koopman & Harvey (2003) algorithm using Kalman gain.
BCI weights = linear combination of indicator factor-weights scaled by GDP factor loadings.

### News Decomposition (Section 4.4, eqs. 23-28)
```
E[BCI_t | Ω_{v+1}] = E[BCI_t | Ω_v] + E[BCI_t | I_{v+1}]
                       (old estimate)      (revision from news)
```
News content: `I_{v+1,j} = y_{i_j, t_j} - E[y_{i_j, t_j} | Ω_v]`
Revision: `E[BCI_t | I_{v+1}] = B_{v+1} I_{v+1} = Σ_j b_{v+1,j} (y_{i_j,t_j} - E[y_{i_j,t_j} | Ω_v])`

---

## Key Implementation Notes

1. **Missing value handling**: set `y_{i,t} = 0` and zero out corresponding row in `H_t`; set corresponding diagonal of `R` to small constant `c` (for invertibility)
2. **Balanced sample for PCA**: minimum 15 years, monthly variables only
3. **Quarterly variance correction**: `V(u_t) = (9/3) V(ũ_t)` for quarterly variables because of G(L) aggregation
4. **Numerical stability**: replace upper-left `n^Q × n^Q` block of `R` with `c I_{n^Q}` in non-quarter months (R must be invertible for Kalman filter)
5. **Aggregation rule choice**: use G(L) = 1/3 + 1/3L + 1/3L² (Angelini et al. 2011), NOT G'(L) = 1/3 + 2/3L + L² + 2/3L³ + 1/3L⁴ (Mariano-Murasawa 2003) — the latter was tested and found inferior for Switzerland
6. **Seasonal adjustment**: apply X-13ARIMA-SEATS to any series not already seasonally adjusted
7. **State dimension**: 3r = 12 (with r=4, p=1)
8. **Key references for implementation**:
   - Kalman filter/smoother: Durbin & Koopman (2012), *Time series analysis by state space models*
   - Two-step estimation: Giannone, Reichlin, Small (2008); Doz, Giannone, Reichlin (2011)
   - Observation weights: Koopman & Harvey (2003)
   - News decomposition: Banbura & Modugno (2014)

---

## Code Structure

The implementation lives in the `mfdfm/` Python package with tests in `tests/` and a runnable demo in `examples/`.

### Package layout

```
mfdfm/
├── __init__.py          # Public API — exports the MFDFM class
├── data.py              # MFData: mixed-frequency data container
├── estimation.py        # Two-step parameter estimation (PCA + OLS + VAR)
├── kalman.py            # General-purpose Kalman filter & RTS smoother
├── model.py             # MFDFM class — main entry point
└── decomposition.py     # Observation weights & news decomposition

tests/
├── conftest.py          # Synthetic data generator fixture
└── test_model.py        # 25 unit/integration tests (unittest)

examples/
└── demo.py              # End-to-end demo with plots (generates mfdfm_demo.png)
```

### Module responsibilities

| Module | Role |
|---|---|
| `data.py` | `MFData` class. Reorders variables as [GDP, quarterly, monthly], standardizes (zero-mean, unit-variance), enforces quarterly NaN at non-quarter months, finds the balanced monthly panel for PCA via greedy variable-dropping. |
| `estimation.py` | **Step 1** of Galli (2017). `estimate_pca_factors` runs PCA on the balanced panel. `estimate_loadings` runs OLS for all variables — monthly vars regress on f^PC directly, quarterly vars regress on G(L)-aggregated factors at quarter-end months with the 9/3 variance correction. `estimate_var` fits VAR(p) on the PC factors. `select_n_factors` / `select_n_lags` implement BIC selection (eqs. 10-11). |
| `kalman.py` | Self-contained Kalman filter and RTS smoother. Accepts time-varying H_t and R_t via callables. Uses the **select-observed** approach: at each t only rows with non-NaN data participate in the update (equivalent to the zero-out + small-c method but cleaner). Uses **Joseph form** for covariance updates and `np.linalg.solve` instead of explicit matrix inversion for numerical stability. Stores per-step innovations, gains, and covariances for downstream decomposition. |
| `model.py` | `MFDFM` class. Orchestrates the full pipeline: builds the companion-form transition (F, Q) with state ξ = [f_t; f_{t-1}; f_{t-2}], constructs the measurement equation (H with 1/3 averaging for quarterly rows, R with 1/3·σ² for quarterly and σ² for monthly), initializes via discrete Lyapunov equation, runs the Kalman smoother, and exposes results. Public API: `fit()`, `business_cycle_index`, `bci_variance`, `bci_accuracy`, `smoothed_factors`, `fitted_values()`, `predict()`, `observation_weights()`, `news_decomposition()`, `summary()`, `save()` / `load()`. Supports `n_factors="auto"` and `n_lags="auto"` for BIC-based selection. |
| `decomposition.py` | Observation weights via Koopman & Harvey (2003) filtered-weight algorithm using stored Kalman gains. News decomposition via Banbura & Modugno (2014): identifies new data releases between two vintages, computes news content I_j = y - E[y\|Ω_v], and decomposes BCI revision into per-indicator contributions. |

### How to run

```bash
# Tests (25 tests, ~8 seconds)
python3 -m unittest tests.test_model -v

# Demo (generates mfdfm_demo.png and prints model summary)
python3 examples/demo.py
```

### Key design decisions

- **Select-observed Kalman filter**: Instead of zeroing H rows and padding R with a small constant c for unobserved variables (paper's approach), unobserved rows are excluded from the update entirely. Mathematically equivalent, avoids invertibility hacks, and faster for large n with many missing values.
- **Greedy balanced panel**: The PCA balanced panel finder iteratively drops the variable that most constrains panel length, until a contiguous block of ≥ 15 years is found. Handles ragged starts robustly.
- **No autocorrelated idiosyncratic errors**: Following the paper's finding that Θ ≠ 0 worsens performance, u_t is i.i.d.
- **Sign convention**: Factor 1 is signed so that the GDP loading is positive.
- **Data preprocessing is external**: The model assumes input data is already seasonally adjusted, transformed to stationarity (3-month diffs / QoQ growth rates), and at monthly frequency. Seasonal adjustment (X-13ARIMA-SEATS) and stationarity transforms are the caller's responsibility.
