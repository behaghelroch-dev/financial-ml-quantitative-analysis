# 📊 Financial Markets — Machine Learning & Quantitative Analysis

> End-to-end project applying Machine Learning and quantitative finance methods to real market data (S&P 500 ETF & Bitcoin), covering classification, time series modeling, temporal ML, and unsupervised market regime detection.

---

## 🧠 Project Overview

This project is structured as a progressive exploration of ML applied to financial markets, across **4 independent notebooks** covering supervised and unsupervised learning, time series econometrics, and rigorous validation methodology.

| Notebook | Focus | Asset | Methods |
|---|---|---|---|
| `TP1` | Market direction classification | SPY (S&P 500) | Logistic Regression, Random Forest, Neural Network |
| `TP2` | Time series econometrics | SPY (S&P 500) | ARIMA, GARCH(1,1) |
| `TP3` | Temporal ML & validation | BTC-USD (Bitcoin) | Random Forest, MLP, Rolling Validation |
| `TP4` | Unsupervised market regime detection | BTC-USD (Bitcoin) | K-Means, DBSCAN |

---

## 📁 Repository Structure

```
├── TP1_market_direction_classification.ipynb
├── TP2_time_series_arima_garch.ipynb
├── TP3_temporal_ml_validation.ipynb
├── TP4_clustering_market_regimes.ipynb
└── README.md
```

---

## TP1 — Market Direction Classification (SPY, 2015–2023)

**Question:** Can we predict whether the S&P 500 will close higher or lower tomorrow?

### Pipeline

**Feature Engineering**
- Daily returns, 5-day rolling volatility, moving average, momentum
- Volume change rate
- Polynomial interaction features: `return × volatility`, `momentum × volatility`, `return × volume_change` (ϕ(x) mapping)

**Chronological Train/Test Split**
- Strict 80/20 temporal split — random shuffle explicitly avoided to prevent look-ahead bias

**Models Compared**

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | ~51% |
| Logistic Regression + interactions | ~52% |
| Random Forest (300 trees, depth=5) | ~53% |
| Neural Network (64→32→1, ReLU + Sigmoid) | ~52% |

**Key Findings**
- All models plateau around 50–55% — consistent with the Efficient Market Hypothesis
- The hard ceiling is the data itself: daily returns have near-zero autocorrelation (ρ(k) ≈ 0)
- Accuracy alone is misleading on imbalanced classes — precision/recall/F1 are also evaluated
- Feature interactions marginally improve the linear model by enabling non-linear separation

---

## TP2 — Time Series Econometrics: ARIMA & GARCH (SPY, 2015–2024)

**Question:** What can we actually model in a financial time series — and where do models hit their limits?

### Analysis Steps

**Stationarity**
- ADF test on prices: p ≈ 0.90 → non-stationary
- ADF test on returns: p ≈ 1e-27 → stationary
- Motivates differencing (d=1 in ARIMA) and working with returns

**ARIMA Modeling**
- Autocorrelation (ACF/PACF) analysis to select (p, d, q) parameters
- Ljung-Box test on residuals confirms remaining ARCH effects → motivates GARCH
- Returns have near-zero autocorrelation → ARIMA finds little exploitable signal

**GARCH(1,1) Modeling**

| Parameter | Value | Interpretation |
|---|---|---|
| ω (omega) | baseline variance | long-run unconditional variance |
| α (alpha) | ≈ 0.20 | reactivity to recent shocks |
| β (beta) | ≈ 0.78 | volatility persistence |
| α + β | ≈ 0.975 | very high persistence — shocks dissipate slowly |

- Peak conditional volatility: **March 2020 (COVID crash)**
- GARCH clearly captures volatility clustering: high vol follows high vol

**Core Insight**
> Forecasting **market direction** (ARIMA) is near-impossible. Forecasting **risk/volatility** (GARCH) is substantially more realistic — volatility has memory, returns do not.

---

## TP3 — Temporal ML & Validation Methodology (BTC-USD, 2018–2024)

**Question:** Can ML models reliably predict Bitcoin returns — and how do we know if a good score is real?

### Features Built
- Lagged returns: `lag_1`, `lag_2`, `lag_5`
- Rolling mean and rolling volatility (5-day window)
- Momentum (price difference over 5 days)
- Volume change rate

### Models Compared

| Model | Notes |
|---|---|
| ARIMA | Econometric baseline |
| Random Forest (300 trees) | Best feature importance interpretability |
| MLP (64→32, ReLU, early stopping) | Scaler fitted on train only — no data leakage |

### Key Methodological Contributions

**Temporal vs. Random Split — Illustrated Danger**
- Random split (`shuffle=True`) introduces look-ahead bias: future data leaks into training
- Demonstrated quantitatively: random split artificially deflates MSE vs. temporal split
- Rule: in time series, test must always be chronologically after train

**Rolling Validation**
- Single train/test split evaluated over one arbitrary time window → unstable estimate
- Rolling window validation (train=400 days, test=50 days, step=50) across the full dataset
- Reveals performance variability: model is systematically worse during high-volatility regimes
- Provides mean + std of MSE across windows → far more honest evaluation

**Feature Importance (Random Forest)**
- `rolling_vol_5` and `momentum_5` are consistently the most predictive features
- Consistent with GARCH findings: volatility has memory, momentum is documented on crypto
- `lag_1`, `lag_2` contribute marginally (low autocorrelation of returns)

---

## TP4 — Unsupervised Market Regime Detection (BTC-USD, 2018–2024)

**Question:** Can we automatically identify distinct market regimes (bull, bear, crisis) without labels?

### Features Built
- 30-day rolling volatility (`volatility_30d`)
- 30-day cumulative return (`return_30d`)
- 30-day momentum (`momentum_30d`)
- Volume change rate

### Normalization
- StandardScaler fitted on full dataset
- Demonstrated that without scaling, `momentum_30d` (expressed in USD, orders of magnitude larger) completely dominates Euclidean distances → clusters become meaningless

### K-Means Clustering

**Optimal K Selection**
- Elbow method: inertia curve with diminishing returns after K=3
- Silhouette score: maximized at K=2 or K=3

**Identified Market Regimes**

| Cluster | Label | Characteristics |
|---|---|---|
| 0 | Calm Market | Low volatility, neutral return |
| 1 | Bull Market | Positive return, moderate volatility |
| 2 | Volatile/Bear Market | High volatility, negative or unstable return |

- Clusters visualized over Bitcoin price history: regime transitions clearly visible at March 2020, late 2021, 2022

### DBSCAN Clustering
- Compared to K-Means: DBSCAN does not require K, and identifies **outliers** (label = -1)
- Outlier points correspond to exceptional market events: COVID crash, FTX collapse, extreme speculation
- Sensitivity to `eps` illustrated: eps=0.2 → mostly noise; eps=0.5 → interpretable clusters; eps=1.0 → everything merges
- K-Means vs. DBSCAN tradeoff: K-Means more readable for defined regimes; DBSCAN better for anomaly detection

---

## 🛠️ Stack

```
Python 3.x
yfinance · pandas · numpy · matplotlib · seaborn
scikit-learn (RandomForestClassifier, RandomForestRegressor, MLPRegressor,
              KMeans, DBSCAN, StandardScaler, silhouette_score)
statsmodels (ARIMA, ADF test, ACF/PACF)
arch (GARCH)
tensorflow / keras (Dense, Sequential — TP1 neural network)
```

---

## 📌 Core Concepts Covered

- **Data leakage prevention**: temporal split, scaler fitted on train only, no look-ahead in features
- **Feature engineering**: lagged returns, rolling statistics, momentum, interaction terms (ϕ(x))
- **Stationarity**: ADF test, differencing, price vs. return analysis
- **Autocorrelation**: ACF/PACF, Ljung-Box test, ARCH effects
- **Econometric modeling**: ARIMA (conditional mean) vs. GARCH (conditional variance)
- **Volatility clustering** and regime shifts (COVID 2020, FTX 2022)
- **Model validation**: single split limitations, rolling window validation, performance distribution
- **Unsupervised learning**: K-Means (elbow + silhouette), DBSCAN, normalization impact
- **Critical analysis**: model assumptions, overfitting risk in low-signal environments

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. All results consistently demonstrate the near-impossibility of reliably predicting short-term market direction, which is itself a key finding. No investment decisions should be based on this work.
