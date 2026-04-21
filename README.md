# Market Risk Intelligence Studio

Market Risk Intelligence Studio is a Streamlit-based machine learning application for forecasting forward-looking market risk from OHLCV data. It is designed to be understandable in an interview setting, demonstrable as a capstone project, and practical to deploy.

## What the project does

- accepts stock or crypto OHLCV CSV uploads
- auto-detects multiple schema styles
- engineers rolling return, volatility, momentum, drawdown, and volume features
- predicts `future_volatility_20d_pct`, the annualized realized volatility over the next 20 trading days
- surfaces EDA, model diagnostics, latest risk rankings, and ticker-level inspection in a polished dashboard

## Why this target was chosen

Most uploaded market files do not contain a labeled risk target. Instead of inventing one manually, the app derives a forward-looking target directly from the time series:

- `future_volatility_20d_pct`

This target is:

- measurable from raw OHLCV data
- financially defensible as a short-horizon risk proxy
- suitable for supervised learning

## Accepted input schemas

The loader auto-normalizes these formats:

### 1. Multi-header panel format

- row 1: `Close`, `High`, `Low`, `Open`, `Volume`
- row 2: ticker symbols
- first column: `Date`

### 2. Flat market format with ticker column

```csv
Date,Symbol,Open,High,Low,Close,Volume
2024-01-01,BTC-USD,42000,42500,41800,42100,25000000
2024-01-02,BTC-USD,42100,43000,42050,42800,27000000
2024-01-01,ETH-USD,2200,2230,2180,2210,18000000
```

Accepted ticker-like fields include:

- `Symbol`
- `Ticker`
- `Asset`
- `Coin`
- `Pair`

### 3. Flat single-asset OHLCV format

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,42000,42500,41800,42100,25000000
2024-01-02,42100,43000,42050,42800,27000000
```

## Modeling approach

The training pipeline uses:

- median imputation with `SimpleImputer`
- `ExtraTreesRegressor` for nonlinear tabular modeling
- time-based train/test splitting instead of random splitting
- a five-year modeling window for recent market relevance

Engineered features include:

- close, open, high, low, volume
- 1-day, 5-day, and 20-day returns
- 20-day and 60-day realized volatility
- 20-day and 60-day momentum
- intraday range percentage
- open-close gap percentage
- 20-day volume ratio
- 60-day drawdown

## Product features

- upload-first workflow for deployment safety
- clear empty state when no local dataset is bundled
- download template for flat market CSV input
- EDA overview with risk-band summaries
- feature importance and prediction-quality views
- latest cross-sectional risk ranking
- ticker explorer for point-in-time diagnostics

## Deployment readiness

This repository is set up for local runs and container deployment.

Included deployment artifacts:

- `Dockerfile`
- `.dockerignore`
- `.streamlit/config.toml`

The large raw dataset is intentionally excluded from git, so the deployed app works in upload-first mode rather than failing on missing local data.

## Project structure

```text
.
├── app.py
├── model_utils.py
├── requirements.txt
├── tests/
│   └── test_model_utils.py
├── .streamlit/
│   └── config.toml
├── Dockerfile
└── data/
    └── sample_portfolio_risk.csv
```

## Local setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python3 -m streamlit run app.py
```

### 4. Open it in the browser

Usually:

```text
http://localhost:8501
```

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

## Docker run

```bash
docker build -t market-risk-intelligence .
docker run -p 8501:8501 market-risk-intelligence
```

## Notes for interview discussion

Good talking points:

- why a forward-engineered target was necessary
- why time-based splitting matters for financial data
- how schema auto-detection improves usability
- why the app was changed to upload-first mode for deployment reliability
- tradeoffs of tree ensembles versus sequence models for tabular market features

## Next possible upgrades

- persist trained models with `joblib`
- add ML experiment tracking
- compare Extra Trees with gradient boosting baselines
- add sector, benchmark, or macro features
- add CI with GitHub Actions
