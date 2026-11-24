"""Score validation.csv and the December chart rows with the trained model.

Run:  python predict.py      (after python train.py)

Writes
  validation_predictions.csv          load_id,predicted_rate for all 12,000 loads
  data/december_chart_inputs.csv       same file with predicted_rate filled in
  reports/december_inputs_used.csv     the exact features fed to the model per day
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight.data import (  # noqa: E402
    DECEMBER_PATH, ROOT, TEMPLATE_PATH, TRAIN_PATH, VALIDATION_PATH,
    city_coordinates, clean_features, daily_market_index, flag_bad_labels, load_raw,
)
from freight.features import build_features, to_rate  # noqa: E402


def main():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    model, features = bundle["model"], bundle["features"]

    train_raw = load_raw(TRAIN_PATH)
    _, weight_medians = clean_features(train_raw[~flag_bad_labels(train_raw)])

    # ---- validation.csv -------------------------------------------------
    val_raw = load_raw(VALIDATION_PATH)
    val, _ = clean_features(val_raw, weight_medians)
    rate = to_rate(model.predict(build_features(val)[features]), val)
    preds = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(rate, 2)})

    template = pd.read_csv(TEMPLATE_PATH)
    out = template[["load_id"]].merge(preds, on="load_id", how="left", validate="one_to_one")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"validation_predictions.csv: {len(out):,} rows, "
          f"median ${out.predicted_rate.median():,.0f}, range ${out.predicted_rate.min():,.0f}-${out.predicted_rate.max():,.0f}")

    # ---- December chart ---------------------------------------------------
    # The chart rows hold lane, distance, equipment, weight and date only.
    # Coordinates come from the city lookup. market_index, the one input that
    # moves with the calendar, is that day's median across the 12,000
    # validation loads, which cover every day of December 2025. Only features
    # are borrowed from validation.csv; it has no labels.
    december = pd.read_csv(DECEMBER_PATH)
    columns = list(december.columns)
    dec = december.copy()
    dec["date"] = pd.to_datetime(dec["date"])
    coords = city_coordinates(train_raw, val_raw)
    for side in ("pickup", "delivery"):
        dec[f"{side}_lat"] = dec[side].map(coords["lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["lon"])
    dec["market_index"] = dec["date"].map(daily_market_index(val_raw))
    dec["weight_missing"] = 0
    assert dec[["pickup_lat", "delivery_lat", "market_index"]].notna().all().all()

    dec_features = build_features(dec)[features]
    december["predicted_rate"] = np.round(to_rate(model.predict(dec_features), dec), 2)
    december[columns].to_csv(DECEMBER_PATH, index=False)
    dec_features.assign(date=dec["date"].dt.date, predicted_rate=december["predicted_rate"]).to_csv(
        ROOT / "reports" / "december_inputs_used.csv", index=False
    )
    print(f"december_chart_inputs.csv: 31 rows, ${december.predicted_rate.min():,.2f}-${december.predicted_rate.max():,.2f}")


if __name__ == "__main__":
    main()
