# Market Risk Intelligence Studio

This project is a capstone-style machine learning application that trains on a real S&P 500 OHLCV dataset and predicts a forward-looking market risk target using Streamlit for local visualization.

## What changed

This version no longer depends on the earlier synthetic sample portfolio table.

It now uses your real market-history dataset:

- raw file: `data/raw/snp500_all_assets.csv`
- source shape: multi-header OHLCV panel
- universe: 503 tickers
- date range: 2010-01-04 to 2023-07-06

## Target used for training

Your raw CSV does not contain an explicit labeled risk column such as `amount_at_risk`.

So the project engineers a valid supervised target from the market history:

- `future_volatility_20d_pct`

This means:

- the annualized realized volatility over the next 20 trading days
- expressed as a percentage
- treated as the model's risk target

This is a reasonable financial risk proxy because higher forward volatility implies higher near-term uncertainty and higher capital exposure.

## Features engineered from the raw data

The model is trained using:

- close
- open
- high
- low
- volume
- 1-day return
- 5-day return
- 20-day return
- 20-day realized volatility
- 60-day realized volatility
- 20-day momentum
- 60-day momentum
- intraday range percentage
- open-close gap percentage
- 20-day volume ratio
- 60-day drawdown

## Model choice

The training pipeline uses:

- `SimpleImputer(strategy="median")`
- `ExtraTreesRegressor`

It also uses:

- a time-based split instead of a random split
- the most recent 5 years of engineered observations
- sampled train/test rows for practical local performance

## Current training snapshot

From the latest run on your dataset:

- train rows used: `200,000`
- test rows used: `50,000`
- split date: `2022-06-06`
- latest modeled date: `2023-06-06`
- MAE: about `7.95%`
- RMSE: about `11.25%`
- R²: about `0.296`

## Dashboard sections

- `EDA Overview`
- `Model Results`
- `Latest Risk Rankings`
- `Ticker Explorer`

## How amount at risk is shown

Because the source dataset contains prices but not your actual portfolio holdings, the dashboard estimates dollar risk using:

`predicted_future_volatility_20d_pct * assumed_position_value`

You can change the assumed position size from the Streamlit sidebar.

## How to run locally

### 1. Go to the project

```bash
cd "/Users/agrwallaadit/risk engine O"
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

If `.venv` does not exist yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the app

```bash
python3 -m streamlit run app.py
```

### 4. Open the browser

Usually:

```text
http://localhost:8501
```

## Accepted upload formats

The app now accepts multiple schema styles and converts them into one internal market-data format automatically.

### 1. Multi-header panel format

This is the original stock-market panel style:

- row 1: `Close`, `High`, `Low`, `Open`, `Volume`
- row 2: ticker symbols
- first column: `Date`

### 2. Flat market format with ticker column

This is common for crypto and exchange exports:

```csv
Date,Symbol,Open,High,Low,Close,Volume
2024-01-01,BTC-USD,42000,42500,41800,42100,25000000
2024-01-02,BTC-USD,42100,43000,42050,42800,27000000
2024-01-01,ETH-USD,2200,2230,2180,2210,18000000
```

Accepted ticker-like column names include:

- `Symbol`
- `Ticker`
- `Asset`
- `Coin`
- `Pair`

### 3. Flat single-asset OHLCV format

This also works:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,42000,42500,41800,42100,25000000
2024-01-02,42100,43000,42050,42800,27000000
```

If no ticker column is present, the app assigns a default asset id internally.

## Required columns for flat files

For flat CSV uploads, the app needs:

- one date column: `Date`, `Datetime`, `Timestamp`, or `Time`
- OHLCV columns: `Open`, `High`, `Low`, `Close`, `Volume`

It can also detect some common variants such as:

- `Adj Close`
- `Adjusted Close`
- `Price`
- `Vol`

## Project files

```text
.
├── app.py
├── model_utils.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── data/
    └── raw/
        └── snp500_all_assets.csv
```

## Future upgrades

- save the trained model with `joblib`
- add notebook-based EDA and reporting
- compare Extra Trees with XGBoost or LightGBM
- add sector metadata and macro features
- use real portfolio weights instead of assumed equal position values
