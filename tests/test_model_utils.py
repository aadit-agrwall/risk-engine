from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_utils import TARGET_COLUMN, load_market_panel, train_model


class ModelUtilsTestCase(unittest.TestCase):
    def _write_temp_csv(self, dataframe: pd.DataFrame) -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        path = temp_dir / "market.csv"
        dataframe.to_csv(path, index=False)
        return path

    def _make_flat_market_frame(self, periods: int = 160) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=periods, freq="D")
        rows = []
        for symbol, base_price in [("BTC-USD", 42000.0), ("ETH-USD", 2200.0)]:
            price = base_price
            for i, current_date in enumerate(dates):
                price = price * (1 + 0.001 + ((i % 7) - 3) / 2000)
                rows.append(
                    {
                        "Date": current_date.strftime("%Y-%m-%d"),
                        "Symbol": symbol,
                        "Open": price * 0.998,
                        "High": price * 1.010,
                        "Low": price * 0.990,
                        "Close": price,
                        "Volume": 1_000_000 + (i * 2500),
                    }
                )
        return pd.DataFrame(rows)

    def test_load_market_panel_accepts_flat_schema(self) -> None:
        path = self._write_temp_csv(self._make_flat_market_frame())
        panel = load_market_panel(path)
        self.assertEqual(
            set(panel.columns.get_level_values(0)),
            {"Close", "High", "Low", "Open", "Volume"},
        )
        self.assertEqual(
            set(panel.columns.get_level_values(1)),
            {"BTC-USD", "ETH-USD"},
        )

    def test_train_model_runs_on_flat_schema(self) -> None:
        path = self._write_temp_csv(self._make_flat_market_frame())
        artifacts = train_model(path)
        self.assertIn("r2", artifacts.metrics)
        self.assertFalse(artifacts.latest_snapshot.empty)
        self.assertIn(TARGET_COLUMN, artifacts.modeling_frame.columns)

    def test_train_model_rejects_short_dataset(self) -> None:
        short_frame = self._make_flat_market_frame(periods=10)
        path = self._write_temp_csv(short_frame)
        with self.assertRaises(ValueError):
            train_model(path)


if __name__ == "__main__":
    unittest.main()
