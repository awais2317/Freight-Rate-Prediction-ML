"""Exploratory analysis: writes figures + a JSON summary to reports/.

Run:  python eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight.data import (  # noqa: E402
    ROOT, TRAIN_PATH, VALIDATION_PATH, flag_bad_labels, load_raw, describe_quality, quote_signal_check,
)

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, ACCENT, MUTED, BAD = "#1F2D33", "#064A56", "#8AA3A8", "#C2410C"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3EAEC", linewidth=0.8)


def main():
    train, val = load_raw(TRAIN_PATH), load_raw(VALIDATION_PATH)
    train["rate_per_mile"] = train["posted_rate"] / train["distance"]
    bad = flag_bad_labels(train)

    # 1. The leakage picture: quote_signal vs distance in each file.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, dpi=150)
    for ax, frame, name in [(axes[0], train, "train_test.csv (Jan 2025)"), (axes[1], val, "validation.csv (Nov-Dec 2025)")]:
        for eq, color in [("Dry Van", ACCENT), ("Reefer", BAD), ("Flatbed", MUTED)]:
            s = frame[frame.equipment == eq].sample(frac=0.35, random_state=0)
            ax.scatter(s.distance, s.quote_signal, s=5, alpha=0.45, color=color, label=eq)
        style(ax, f"quote_signal vs distance: {name}")
        ax.set_xlabel("distance (miles)")
    axes[0].set_ylabel("quote_signal")
    axes[0].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / "quote_signal_shift.png")
    plt.close(fig)

    # 2. Label corruption, seen through rate / (quote_signal * distance).
    ratio = train["posted_rate"] / (train["quote_signal"] * train["distance"])
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.hist(np.log10(ratio.dropna()), bins=120, color=ACCENT)
    for x in np.log10([0.9, 1.1]):
        ax.axvline(x, color=BAD, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("log10( posted_rate / (quote_signal x distance) )")
    style(ax, f"Label audit: {int(bad.sum()) - 1} corrupted labels outside the +/-10% band (red)")
    fig.tight_layout()
    fig.savefig(FIG / "label_audit.png")
    plt.close(fig)

    # 3. Rate per mile vs distance on clean rows.
    clean = train[~bad]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    for eq, color in [("Dry Van", ACCENT), ("Reefer", BAD), ("Flatbed", MUTED)]:
        s = clean[clean.equipment == eq]
        ax.scatter(s.distance, s.rate_per_mile, s=5, alpha=0.4, color=color, label=eq)
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("$ per mile")
    ax.legend(frameon=False, markerscale=3)
    style(ax, "Rate per mile falls with distance; Reefer > Flatbed > Dry Van")
    fig.tight_layout()
    fig.savefig(FIG / "rate_per_mile.png")
    plt.close(fig)

    # 4. market_index over time in both files (it is the only date-driven input).
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=150)
    for frame, color, label in [(train, ACCENT, "train"), (val, BAD, "validation")]:
        daily = frame.groupby("date").market_index.median()
        ax.plot(daily.index, daily.values, marker="o", markersize=2.5, color=color, label=label)
    ax.set_ylabel("daily median market_index")
    ax.legend(frameon=False)
    style(ax, "market_index: weekly cycle (weekend lows, Thursday highs), same range in both files")
    fig.tight_layout()
    fig.savefig(FIG / "market_index.png")
    plt.close(fig)

    seen = set(train.pickup) | set(train.delivery)
    new_cities = sorted((set(val.pickup) | set(val.delivery)) - seen)
    summary = {
        "quality": [describe_quality(train, "train_test.csv"), describe_quality(val, "validation.csv")],
        "quote_signal_train": quote_signal_check(train),
        "quote_signal_validation": quote_signal_check(val),
        "label_ratio_clean_std": round(float(ratio[~bad].std()), 4),
        "new_cities_in_validation": new_cities,
        "validation_rows_touching_new_city": int((val.pickup.isin(new_cities) | val.delivery.isin(new_cities)).sum()),
        "clean_rate_per_mile_by_equipment": clean.groupby("equipment").rate_per_mile.median().round(3).to_dict(),
    }
    (ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
