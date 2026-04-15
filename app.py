from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import TARGET_COLUMN, train_model

DEFAULT_DATA_PATH = Path("data/raw/snp500_all_assets.csv")

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

st.set_page_config(
    page_title="Market Risk Intelligence Studio",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(8, 78, 104, 0.17), transparent 26%),
                radial-gradient(circle at top right, rgba(215, 176, 78, 0.22), transparent 24%),
                linear-gradient(180deg, #f7f4ec 0%, #edf3f3 54%, #f8f8f5 100%);
            color: #12212a;
        }
        .block-container {
            max-width: 1300px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            color: #12202a;
            letter-spacing: -0.02em;
        }
        .hero-shell {
            padding: 1.8rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0d3442 0%, #0f5a63 45%, #d0a94c 100%);
            color: #f9f5ea;
            box-shadow: 0 22px 54px rgba(18, 32, 42, 0.18);
            margin-bottom: 1.2rem;
        }
        .hero-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.24em;
            opacity: 0.82;
            margin-bottom: 0.65rem;
        }
        .hero-title {
            font-size: 2.6rem;
            line-height: 1.05;
            margin-bottom: 0.65rem;
            font-weight: 700;
        }
        .hero-copy {
            max-width: 880px;
            font-size: 1.02rem;
            line-height: 1.6;
            opacity: 0.96;
        }
        .mini-card {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(18, 32, 42, 0.08);
            border-radius: 18px;
            padding: 1rem;
            min-height: 126px;
            box-shadow: 0 12px 28px rgba(15, 34, 44, 0.08);
        }
        .mini-label {
            color: #5c6972;
            text-transform: uppercase;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            margin-bottom: 0.55rem;
        }
        .mini-value {
            color: #11202a;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .mini-copy {
            color: #43515a;
            font-size: 0.94rem;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(18, 32, 42, 0.08);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15, 34, 44, 0.08);
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102f3d 0%, #153743 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #eef3f4;
        }
        .section-note {
            background: rgba(255, 255, 255, 0.76);
            border-left: 5px solid #d0a94c;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin: 0.4rem 0 1rem 0;
            color: #33424d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Training the model on the selected market dataset...")
def get_artifacts_from_path(path: str):
    return train_model(path)


@st.cache_resource(show_spinner="Training the model on the uploaded market dataset...")
def get_artifacts_from_upload(file_bytes: bytes):
    return train_model(file_bytes)


def currency(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def risk_band(volatility_pct: pd.Series) -> pd.Series:
    bins = [-1, 20, 30, 45, 1000]
    labels = ["Low", "Moderate", "High", "Severe"]
    return pd.cut(volatility_pct, bins=bins, labels=labels)


def add_risk_columns(dataframe: pd.DataFrame, position_size: float) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["risk_band"] = risk_band(enriched["predicted_future_volatility_20d_pct"]).astype(str)
    enriched["position_size"] = position_size
    enriched["estimated_amount_at_risk"] = (
        enriched["predicted_future_volatility_20d_pct"] / 100.0
    ) * position_size
    return enriched


def make_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
            <div class="mini-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Real Dataset Capstone Build</div>
        <div class="hero-title">Market Risk Intelligence Studio</div>
        <div class="hero-copy">
            This version is trained on your S&amp;P 500 market dataset, not on synthetic demo rows.
            It reshapes the multi-header OHLCV panel, engineers predictive features, and learns forward
            20-trading-day realized volatility as the risk target.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if px is None:
    st.warning(
        "Plotly is not installed in this environment. The dashboard still works, but charts fall back to simpler Streamlit visuals. "
        "Install it with: pip install plotly"
    )

with st.sidebar:
    st.header("Control Center")
    uploaded_file = st.file_uploader("Upload market CSV", type=["csv"])
    position_size = st.number_input(
        "Assumed position value per asset ($)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0,
    )
    latest_count = st.slider("Top assets to display", min_value=10, max_value=50, value=20, step=5)
    st.caption(
        "The amount-at-risk figure uses the predicted future volatility percentage times this assumed position value."
    )

if uploaded_file is not None:
    artifacts = get_artifacts_from_upload(uploaded_file.getvalue())
    source_label = "Uploaded market dataset"
else:
    artifacts = get_artifacts_from_path(str(DEFAULT_DATA_PATH))
    source_label = "Bundled S&P 500 market dataset"

latest_snapshot = add_risk_columns(artifacts.latest_snapshot, position_size)
top_latest = latest_snapshot.head(latest_count).copy()

portfolio_risk_total = float(top_latest["estimated_amount_at_risk"].sum())
avg_predicted_risk = float(latest_snapshot["predicted_future_volatility_20d_pct"].mean())
severe_assets = int((latest_snapshot["risk_band"] == "Severe").sum())
highest_asset = latest_snapshot.iloc[0]

st.markdown(
    """
    <div class="section-note">
        Target definition: <b>future_volatility_20d_pct</b> is the annualized realized volatility over the next 20 trading days.
        Because your source file did not contain a labeled risk column, this target was engineered directly from the price history
        so the model can learn a forward-looking risk estimate from real market behavior.
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2, top3 = st.columns(3)
with top1:
    make_card(
        "Dataset source",
        source_label,
        "The app trains on the selected market panel and uses the most recent date for live cross-sectional risk ranking.",
    )
with top2:
    make_card(
        "Training design",
        "Time-based split",
        f"Only the last {artifacts.split_summary['lookback_years']} years are modeled, with training ending on {artifacts.split_summary['split_date']}.",
    )
with top3:
    make_card(
        "Risk translation",
        "Volatility to dollars",
        "Estimated amount at risk is computed as predicted forward volatility percent times the assumed position value per asset.",
    )

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Tickers", f"{artifacts.raw_summary['ticker_count']:,}")
metric2.metric("Trading days", f"{artifacts.raw_summary['trading_days']:,}")
metric3.metric("Latest modeled date", artifacts.split_summary["latest_date"])
metric4.metric("Average predicted 20d risk", pct(avg_predicted_risk))

metric5, metric6, metric7, metric8 = st.columns(4)
metric5.metric("Model R²", f"{artifacts.metrics['r2']:.3f}")
metric6.metric("MAE", pct(artifacts.metrics["mae"]))
metric7.metric("Top basket amount at risk", currency(portfolio_risk_total))
metric8.metric("Severe-risk assets", f"{severe_assets}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "EDA Overview",
        "Model Results",
        "Latest Risk Rankings",
        "Ticker Explorer",
    ]
)

with tab1:
    st.subheader("Exploratory Data Analysis")
    eda1, eda2, eda3 = st.columns(3)
    eda1.metric("Date range", f"{artifacts.raw_summary['start_date']} to {artifacts.raw_summary['end_date']}")
    eda2.metric("Avg missing close", pct(artifacts.raw_summary["avg_missing_close_pct"]))
    eda3.metric("Median daily return", pct(artifacts.raw_summary["median_daily_return_pct"]))

    left, right = st.columns(2)
    with left:
        target_dist = artifacts.modeling_frame[TARGET_COLUMN]
        if px is not None:
            fig = px.histogram(
                artifacts.modeling_frame.sample(min(len(artifacts.modeling_frame), 50000), random_state=42),
                x=TARGET_COLUMN,
                nbins=60,
                title="Engineered target distribution: future 20-day volatility",
                color_discrete_sequence=["#0c3b4c"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(target_dist.value_counts(bins=40, sort=False))

    with right:
        recent_summary = (
            latest_snapshot.groupby("risk_band", dropna=False)
            .agg(
                asset_count=("ticker", "count"),
                avg_predicted_risk=("predicted_future_volatility_20d_pct", "mean"),
            )
            .reset_index()
        )
        if px is not None:
            fig = px.bar(
                recent_summary,
                x="risk_band",
                y="asset_count",
                color="avg_predicted_risk",
                color_continuous_scale=["#7fb069", "#d0a94c", "#a63d40"],
                title="Risk-band distribution on the latest modeled date",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(recent_summary.set_index("risk_band")["asset_count"])

    st.dataframe(
        pd.DataFrame(
            [
                {"metric": "Modeling rows", "value": f"{artifacts.split_summary['modeling_rows']:,}"},
                {"metric": "Train rows used", "value": f"{artifacts.split_summary['train_rows']:,}"},
                {"metric": "Test rows used", "value": f"{artifacts.split_summary['test_rows']:,}"},
                {"metric": "Target mean", "value": pct(artifacts.split_summary["target_mean_pct"])},
                {"metric": "Target median", "value": pct(artifacts.split_summary["target_median_pct"])},
                {"metric": "Target 95th percentile", "value": pct(artifacts.split_summary["target_p95_pct"])},
                {"metric": "Latest highest-risk ticker", "value": highest_asset["ticker"]},
                {"metric": "Latest highest predicted risk", "value": pct(float(highest_asset["predicted_future_volatility_20d_pct"]))},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Model Performance and Explainability")
    left, right = st.columns(2)

    with left:
        if px is not None:
            fig = px.bar(
                artifacts.feature_importance.head(10),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#d0a94c", "#0c3b4c"],
                title="Top feature drivers",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(artifacts.feature_importance.head(10).set_index("feature")["importance"])

    with right:
        sample_predictions = artifacts.test_predictions.sample(
            min(len(artifacts.test_predictions), 12000), random_state=42
        )
        if px is not None:
            fig = px.scatter(
                sample_predictions,
                x=TARGET_COLUMN,
                y="predicted_future_volatility_20d_pct",
                color="volatility_20d",
                hover_name="ticker",
                color_continuous_scale=["#7fb069", "#d0a94c", "#a63d40"],
                title="Actual vs predicted forward volatility",
            )
            lower = float(sample_predictions[TARGET_COLUMN].min())
            upper = float(sample_predictions[TARGET_COLUMN].max())
            fig.add_shape(
                type="line",
                x0=lower,
                y0=lower,
                x1=upper,
                y1=upper,
                line={"dash": "dash", "color": "#12212a"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(
                sample_predictions.rename(
                    columns={
                        TARGET_COLUMN: "Actual",
                        "predicted_future_volatility_20d_pct": "Predicted",
                    }
                ),
                x="Actual",
                y="Predicted",
            )

    st.dataframe(
        artifacts.test_predictions.sort_values("prediction_error", key=lambda s: s.abs(), ascending=False).head(100),
        use_container_width=True,
    )

with tab3:
    st.subheader("Latest Risk Rankings")
    left, right = st.columns(2)

    with left:
        if px is not None:
            fig = px.bar(
                top_latest,
                x="predicted_future_volatility_20d_pct",
                y="ticker",
                orientation="h",
                color="risk_band",
                color_discrete_map={
                    "Low": "#7fb069",
                    "Moderate": "#d0a94c",
                    "High": "#d97757",
                    "Severe": "#a63d40",
                },
                title=f"Top {latest_count} predicted-risk tickers on {artifacts.split_summary['latest_date']}",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top_latest.set_index("ticker")["predicted_future_volatility_20d_pct"])

    with right:
        if px is not None:
            fig = px.scatter(
                latest_snapshot.sample(min(len(latest_snapshot), 400), random_state=42),
                x="close",
                y="predicted_future_volatility_20d_pct",
                size="volume_ratio_20d",
                color="risk_band",
                hover_name="ticker",
                color_discrete_map={
                    "Low": "#7fb069",
                    "Moderate": "#d0a94c",
                    "High": "#d97757",
                    "Severe": "#a63d40",
                },
                title="Latest cross-sectional risk map",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(
                latest_snapshot.rename(
                    columns={
                        "close": "Close",
                        "predicted_future_volatility_20d_pct": "Predicted Risk",
                    }
                ),
                x="Close",
                y="Predicted Risk",
            )

    ranking_table = top_latest[
        [
            "ticker",
            "date",
            "close",
            "volatility_20d",
            "volatility_60d",
            "predicted_future_volatility_20d_pct",
            "risk_band",
            "estimated_amount_at_risk",
        ]
    ].copy()
    st.dataframe(ranking_table, use_container_width=True)
    st.download_button(
        "Download latest risk rankings CSV",
        data=latest_snapshot.to_csv(index=False).encode("utf-8"),
        file_name="latest_risk_rankings.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Ticker Explorer")
    available_tickers = sorted(latest_snapshot["ticker"].unique().tolist())
    selected_ticker = st.selectbox("Select ticker", options=available_tickers, index=0)
    ticker_history = artifacts.modeling_frame[artifacts.modeling_frame["ticker"] == selected_ticker].copy()
    ticker_latest = latest_snapshot[latest_snapshot["ticker"] == selected_ticker].iloc[0]

    ex1, ex2, ex3 = st.columns(3)
    ex1.metric("Latest close", currency(float(ticker_latest["close"])))
    ex2.metric("Predicted future 20d risk", pct(float(ticker_latest["predicted_future_volatility_20d_pct"])))
    ex3.metric(
        "Estimated amount at risk",
        currency(float(ticker_latest["estimated_amount_at_risk"] if "estimated_amount_at_risk" in ticker_latest else (ticker_latest["predicted_future_volatility_20d_pct"] / 100.0) * position_size)),
    )

    if px is not None:
        risk_line = px.line(
            ticker_history.tail(260),
            x="date",
            y=["volatility_20d", TARGET_COLUMN],
            title=f"{selected_ticker}: recent realized vs future volatility target",
            labels={"value": "Volatility %", "date": "Date", "variable": "Series"},
        )
        st.plotly_chart(risk_line, use_container_width=True)

        price_line = px.line(
            ticker_history.tail(260),
            x="date",
            y="close",
            title=f"{selected_ticker}: closing price history",
        )
        st.plotly_chart(price_line, use_container_width=True)
    else:
        st.line_chart(
            ticker_history.tail(260).set_index("date")[["volatility_20d", TARGET_COLUMN]]
        )
        st.line_chart(ticker_history.tail(260).set_index("date")["close"])

    st.dataframe(
        ticker_history.tail(60)[
            [
                "date",
                "ticker",
                "close",
                "return_20d",
                "volatility_20d",
                "volatility_60d",
                "momentum_20d",
                "drawdown_60d",
                TARGET_COLUMN,
            ]
        ],
        use_container_width=True,
    )
