"""Loading and cleaning.

Everything that touches raw data quality lives here so the rules are in one
place and are applied the same way to train, validation and the December rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

# A clean label sits within ~1% of quote_signal * distance in the training file
# (ratio mean 1.000, std 0.011). Everything outside this band is a 0.2-0.4x or
# 2.3-5x multiplier, i.e. a corrupted label rather than a genuinely odd load.
LABEL_RATIO_BAND = (0.9, 1.1)


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon). Each city has exactly one coordinate pair across all files."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            if f"{side}_lat" in frame:
                parts.append(
                    frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
                )
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def flag_bad_labels(train: pd.DataFrame) -> pd.Series:
    """True for rows whose posted_rate should not be learned from.

    quote_signal is only trustworthy *inside* the training file (see EDA), so we
    use it here purely as an audit tool for the label, never as a model input.
    """
    ratio = train["posted_rate"] / (train["quote_signal"] * train["distance"])
    lo, hi = LABEL_RATIO_BAND
    bad = train["posted_rate"].isna() | ~ratio.between(lo, hi)
    # The one row with no quote_signal also has no label, so it is caught above.
    return bad


def clean_features(frame: pd.DataFrame, weight_medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Fix feature-level quality issues. Returns the cleaned frame and the
    per-equipment weight medians (fit on train, reused everywhere else)."""
    out = frame.copy()

    # Negative weights have the same magnitude distribution as positive ones
    # (mean |w| 31.0k vs 31.2k), so they are sign flips, not a separate population.
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    if weight_medians is None:
        weight_medians = out.groupby("equipment")["weight"].median()
    out["weight"] = out["weight"].fillna(out["equipment"].map(weight_medians))

    # market_index is a daily market level plus small row noise (within-day
    # sd ~0.025, no relation to geography), so the best fill is that day's median.
    if "market_index" in out:
        out["market_index_missing"] = out["market_index"].isna().astype(int)
        daily = out.groupby("date")["market_index"].transform("median")
        out["market_index"] = out["market_index"].fillna(daily)

    return out, weight_medians


def load_training_data() -> pd.DataFrame:
    """Training rows that are safe to learn from, with cleaned features."""
    raw = load_raw(TRAIN_PATH)
    bad = flag_bad_labels(raw)
    clean, _ = clean_features(raw[~bad])
    return clean.reset_index(drop=True)


def daily_market_index(frame: pd.DataFrame) -> pd.Series:
    """Median market_index per calendar day (used for the December chart)."""
    return frame.groupby("date")["market_index"].median()


def describe_quality(raw: pd.DataFrame, name: str) -> dict:
    w = raw["weight"]
    info = {
        "file": name,
        "rows": int(len(raw)),
        "date_min": str(raw["date"].min().date()),
        "date_max": str(raw["date"].max().date()),
        "missing_weight": int(w.isna().sum()),
        "negative_weight": int((w < 0).sum()),
        "weight_at_cap_47500": int((w.abs() == 47_500).sum()),
        "missing_market_index": int(raw["market_index"].isna().sum()),
        "distance_at_floor_70": int((raw["distance"] == 70).sum()),
    }
    if "posted_rate" in raw:
        info["missing_label"] = int(raw["posted_rate"].isna().sum())
        info["corrupted_label"] = int(flag_bad_labels(raw).sum() - raw["posted_rate"].isna().sum())
    return info


def quote_signal_check(raw: pd.DataFrame) -> dict:
    """How much structure quote_signal carries in a given file."""
    return {
        "corr_with_inverse_distance": round(float(raw["quote_signal"].corr(1 / raw["distance"])), 3),
        "mean_by_equipment": raw.groupby("equipment")["quote_signal"].mean().round(3).to_dict(),
        "mean_short_haul_lt300": round(float(raw.loc[raw.distance < 300, "quote_signal"].mean()), 3),
        "mean_long_haul_gt2000": round(float(raw.loc[raw.distance > 2000, "quote_signal"].mean()), 3),
    }

