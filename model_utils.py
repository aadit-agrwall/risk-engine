from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "future_volatility_20d_pct"
FORWARD_WINDOW = 20
LOOKBACK_YEARS = 5
FEATURE_COLUMNS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volatility_60d",
    "momentum_20d",
    "momentum_60d",
    "range_pct",
    "open_close_gap",
    "volume_ratio_20d",
    "drawdown_60d",
]


@dataclass
class TrainingArtifacts:
    model: Pipeline
    metrics: dict
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame
    latest_snapshot: pd.DataFrame
    modeling_frame: pd.DataFrame
    raw_summary: dict
    split_summary: dict


def load_market_panel(source: str | Path | bytes) -> pd.DataFrame:
    raw_source = BytesIO(source) if isinstance(source, bytes) else source

    try:
        dataframe = pd.read_csv(raw_source, header=[0, 1], index_col=0)
        dataframe.index = pd.to_datetime(dataframe.index)
        expected_fields = {"Close", "Open", "High", "Low", "Volume"}
        actual_fields = set(dataframe.columns.get_level_values(0))
        if actual_fields == expected_fields and not dataframe.index.isna().all():
            return dataframe.sort_index()
    except Exception:
        pass

    raw_source = BytesIO(source) if isinstance(source, bytes) else source
    flat = pd.read_csv(raw_source)
    return flat_to_market_panel(flat)


def flat_to_market_panel(dataframe: pd.DataFrame) -> pd.DataFrame:
    renamed = dataframe.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]

    lower_map = {column.lower(): column for column in renamed.columns}

    def get_column(options: list[str], required: bool = True) -> str | None:
        for option in options:
            if option in lower_map:
                return lower_map[option]
        if required:
            raise ValueError(
                "Could not detect a required column. "
                f"Expected one of: {', '.join(options)}"
            )
        return None

    date_column = get_column(["date", "datetime", "timestamp", "time"])
    open_column = get_column(["open", "open price", "opening price"])
    high_column = get_column(["high", "high price"])
    low_column = get_column(["low", "low price"])
    close_column = get_column(["close", "adj close", "adjusted close", "price"])
    volume_column = get_column(["volume", "vol", "base volume", "quote volume"])
    ticker_column = get_column(["ticker", "symbol", "asset", "coin", "pair"], required=False)

    normalized = renamed.rename(
        columns={
            date_column: "Date",
            open_column: "Open",
            high_column: "High",
            low_column: "Low",
            close_column: "Close",
            volume_column: "Volume",
        }
    ).copy()

    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    if normalized["Date"].isna().all():
        raise ValueError("Could not parse any values in the date column.")

    if ticker_column is None:
        normalized["Ticker"] = "ASSET_1"
    else:
        normalized["Ticker"] = renamed[ticker_column].astype(str).fillna("ASSET_1")

    normalized = normalized.dropna(subset=["Date"]).copy()
    numeric_fields = ["Open", "High", "Low", "Close", "Volume"]
    for field in numeric_fields:
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce")

    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    if normalized.empty:
        raise ValueError(
            "No usable OHLC rows were found after parsing the uploaded file."
        )

    pivoted = {}
    for field in ["Close", "High", "Low", "Open", "Volume"]:
        pivoted[field] = (
            normalized.pivot_table(
                index="Date",
                columns="Ticker",
                values=field,
                aggfunc="last",
            )
            .sort_index()
            .sort_index(axis=1)
        )

    panel = pd.concat(pivoted, axis=1)
    panel.columns.names = [None, None]
    return panel.sort_index()


def panel_to_long(dataframe: pd.DataFrame) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for field in ["Close", "Open", "High", "Low", "Volume"]:
        stacked = dataframe[field].stack().rename(field.lower()).reset_index()
        stacked.columns = ["date", "ticker", field.lower()]
        frames[field.lower()] = stacked

    long_frame = frames["close"]
    for field in ["open", "high", "low", "volume"]:
        long_frame = long_frame.merge(frames[field], on=["date", "ticker"], how="left")

    long_frame = long_frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    long_frame["ticker"] = long_frame["ticker"].astype(str)
    return long_frame


def _forward_rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.iloc[::-1].rolling(window, min_periods=window).std().iloc[::-1].shift(-1)


def engineer_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    engineered = long_frame.copy()
    grouped = engineered.groupby("ticker", group_keys=False)

    engineered["return_1d"] = grouped["close"].pct_change() * 100
    engineered["return_5d"] = grouped["close"].pct_change(5) * 100
    engineered["return_20d"] = grouped["close"].pct_change(20) * 100
    engineered["volatility_20d"] = (
        grouped["close"].pct_change().rolling(20).std().reset_index(level=0, drop=True)
        * np.sqrt(252)
        * 100
    )
    engineered["volatility_60d"] = (
        grouped["close"].pct_change().rolling(60).std().reset_index(level=0, drop=True)
        * np.sqrt(252)
        * 100
    )
    engineered["momentum_20d"] = (
        engineered["close"]
        / grouped["close"].rolling(20).mean().reset_index(level=0, drop=True)
        - 1
    ) * 100
    engineered["momentum_60d"] = (
        engineered["close"]
        / grouped["close"].rolling(60).mean().reset_index(level=0, drop=True)
        - 1
    ) * 100
    engineered["range_pct"] = ((engineered["high"] - engineered["low"]) / engineered["close"]) * 100
    engineered["open_close_gap"] = ((engineered["close"] - engineered["open"]) / engineered["open"]) * 100
    engineered["volume_ratio_20d"] = (
        engineered["volume"]
        / grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    engineered["drawdown_60d"] = (
        engineered["close"]
        / grouped["close"].rolling(60).max().reset_index(level=0, drop=True)
        - 1
    ) * 100
    forward_vol = grouped["close"].pct_change().transform(
        lambda series: _forward_rolling_std(series, FORWARD_WINDOW)
    )
    engineered[TARGET_COLUMN] = forward_vol * np.sqrt(252) * 100
    engineered["month"] = engineered["date"].dt.month
    engineered["weekday"] = engineered["date"].dt.dayofweek

    engineered = engineered.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return engineered


def summarize_raw_panel(dataframe: pd.DataFrame) -> dict:
    close_frame = dataframe["Close"]
    returns = close_frame.pct_change()
    return {
        "ticker_count": int(close_frame.shape[1]),
        "trading_days": int(close_frame.shape[0]),
        "start_date": str(close_frame.index.min().date()),
        "end_date": str(close_frame.index.max().date()),
        "avg_missing_close_pct": float(close_frame.isna().mean().mean() * 100),
        "median_daily_return_pct": float(returns.stack().median() * 100),
        "daily_volatility_pct": float(returns.stack().std() * 100),
    }


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=180,
                    max_depth=18,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(source: str | Path | bytes) -> TrainingArtifacts:
    raw_panel = load_market_panel(source)
    raw_summary = summarize_raw_panel(raw_panel)
    modeling_frame = engineer_features(panel_to_long(raw_panel))

    if modeling_frame.empty:
        raise ValueError(
            "No usable training rows were created from the uploaded file. "
            "This usually means the dataset is too short, has too many missing values, "
            "or does not match the expected OHLCV market-data format."
        )

    cutoff_date = modeling_frame["date"].max() - pd.Timedelta(days=365 * LOOKBACK_YEARS)
    modeling_frame = modeling_frame[modeling_frame["date"] >= cutoff_date].copy()

    if modeling_frame.empty:
        raise ValueError(
            "No rows remain after applying the modeling window. "
            "The dataset needs enough dated price history to compute rolling features "
            "and the forward 20-day volatility target."
        )

    unique_dates = np.sort(modeling_frame["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError(
            "The dataset does not contain enough distinct dates to build a train/test split. "
            "Upload a file with more historical observations."
        )
    split_index = int(len(unique_dates) * 0.8)
    split_date = pd.Timestamp(unique_dates[split_index])
    train_frame = modeling_frame[modeling_frame["date"] <= split_date].copy()
    test_frame = modeling_frame[modeling_frame["date"] > split_date].copy()

    if train_frame.empty or test_frame.empty:
        raise ValueError(
            "The dataset did not produce both training and test samples after preprocessing. "
            "Try a longer historical file with more valid daily records."
        )

    if len(train_frame) > 200_000:
        train_frame = train_frame.sample(200_000, random_state=42)
    if len(test_frame) > 50_000:
        test_frame = test_frame.sample(50_000, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])

    predictions = pipeline.predict(test_frame[FEATURE_COLUMNS])
    prediction_frame = test_frame[
        ["date", "ticker", "close", "volatility_20d", "volatility_60d", TARGET_COLUMN]
    ].copy()
    prediction_frame["predicted_future_volatility_20d_pct"] = predictions
    prediction_frame["prediction_error"] = (
        prediction_frame[TARGET_COLUMN] - prediction_frame["predicted_future_volatility_20d_pct"]
    )

    metrics = {
        "mae": float(mean_absolute_error(test_frame[TARGET_COLUMN], predictions)),
        "rmse": float(root_mean_squared_error(test_frame[TARGET_COLUMN], predictions)),
        "r2": float(r2_score(test_frame[TARGET_COLUMN], predictions)),
    }

    model = pipeline.named_steps["model"]
    feature_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    latest_date = modeling_frame["date"].max()
    latest_snapshot = modeling_frame[modeling_frame["date"] == latest_date].copy()
    latest_snapshot["predicted_future_volatility_20d_pct"] = pipeline.predict(
        latest_snapshot[FEATURE_COLUMNS]
    )
    latest_snapshot = latest_snapshot.sort_values(
        "predicted_future_volatility_20d_pct", ascending=False
    ).reset_index(drop=True)

    split_summary = {
        "lookback_years": LOOKBACK_YEARS,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "split_date": str(split_date.date()),
        "modeling_rows": int(len(modeling_frame)),
        "target_mean_pct": float(modeling_frame[TARGET_COLUMN].mean()),
        "target_median_pct": float(modeling_frame[TARGET_COLUMN].median()),
        "target_p95_pct": float(modeling_frame[TARGET_COLUMN].quantile(0.95)),
        "latest_date": str(latest_date.date()),
    }

    return TrainingArtifacts(
        model=pipeline,
        metrics=metrics,
        feature_importance=feature_importance,
        test_predictions=prediction_frame.reset_index(drop=True),
        latest_snapshot=latest_snapshot,
        modeling_frame=modeling_frame.reset_index(drop=True),
        raw_summary=raw_summary,
        split_summary=split_summary,
    )


def predict_frame(model: Pipeline, dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output["predicted_future_volatility_20d_pct"] = model.predict(output[FEATURE_COLUMNS])
    return output
