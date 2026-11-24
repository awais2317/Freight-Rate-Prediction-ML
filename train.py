"""Validation experiments + final model fit.

Run:  python train.py

Writes
  reports/validation_results.csv / .json   model x split-scheme metrics
  reports/figures/*.png                     holdout + importance charts
  models/model.joblib                        final model (all clean Jan rows)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight.data import ROOT, VALIDATION_PATH, TRAIN_PATH, clean_features, flag_bad_labels, load_raw, load_training_data  # noqa: E402
from freight.features import EXPERIMENTAL, FEATURES, build_features, target  # noqa: E402
from freight.models import candidates, final_model, gradient_boosting  # noqa: E402
from freight.validation import SCHEMES, TEMPORAL_CUTOFF, evaluate, metrics  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MODELS = ROOT / "models"
INK, ACCENT, BAD = "#1F2D33", "#064A56", "#C2410C"


def compare_models(train, X, y) -> pd.DataFrame:
    rows = []
    for name, model in candidates().items():
        for scheme in SCHEMES:
            result = evaluate(model, X[FEATURES], y, train, scheme)
            rows.append({"model": name, "scheme": scheme, **result})
            print(f"  {name:<28} {scheme:<13} MAPE {result['mape_pct']:.2f}%  MAE ${result['mae_usd']:.0f}")
    return pd.DataFrame(rows)


def feature_ablations(train, X, y) -> list[dict]:
    """Things that were tried and rejected, with the evidence."""
    out = []
    base = evaluate(final_model(), X[FEATURES], y, train, "temporal")["mape_pct"]
    for extra in EXPERIMENTAL:
        with_extra = evaluate(final_model(), X[FEATURES + [extra]], y, train, "temporal")["mape_pct"]
        out.append({"change": f"+ {extra}", "temporal_mape_pct": with_extra, "baseline_mape_pct": base})
    return out


def leakage_demo(train, X, y) -> dict:
    """Why quote_signal is excluded.

    Fit with quote_signal on 1-21 Jan, then score the 22-28 Jan holdout twice:
    once as-is, once with quote_signal shuffled across rows, which is how it
    behaves in validation.csv (no relation to distance or equipment there).
    """
    Xq = X[FEATURES].assign(quote_signal=train["quote_signal"].to_numpy())
    is_test = (train["date"] >= TEMPORAL_CUTOFF).to_numpy()
    model = gradient_boosting().fit(Xq[~is_test], y[~is_test])

    test = Xq[is_test]
    shuffled = test.assign(quote_signal=np.random.default_rng(0).permutation(test["quote_signal"].to_numpy()))
    dist = train["distance"].to_numpy()[is_test]
    truth = np.exp(y[is_test].to_numpy()) * dist

    honest = final_model().fit(X[FEATURES][~is_test], y[~is_test])
    return {
        "with_quote_signal_as_in_train_mape_pct": metrics(truth, np.exp(model.predict(test)) * dist)["mape_pct"],
        "with_quote_signal_as_in_validation_mape_pct": metrics(truth, np.exp(model.predict(shuffled)) * dist)["mape_pct"],
        "without_quote_signal_mape_pct": metrics(truth, np.exp(honest.predict(X[FEATURES][is_test])) * dist)["mape_pct"],
    }


def adversarial_check() -> dict:
    """Can a classifier tell train rows from validation rows?

    Done twice: on the features the model uses, and with quote_signal added.
    A high AUC only once quote_signal is added means the shift lives in that
    column, not in the inputs we rely on.
    """
    train_raw, val_raw = load_raw(TRAIN_PATH), load_raw(VALIDATION_PATH)
    train_raw = train_raw[~flag_bad_labels(train_raw)]
    tr, med = clean_features(train_raw)
    va, _ = clean_features(val_raw, med)
    both = pd.concat([tr, va], ignore_index=True)
    is_val = np.r_[np.zeros(len(tr)), np.ones(len(va))]
    X = build_features(both)[FEATURES]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    result = {}
    for label, cols in [("model_features", X), ("model_features_plus_quote_signal", X.assign(qs=both["quote_signal"].to_numpy()))]:
        p = cross_val_predict(clf, cols, is_val, cv=cv, method="predict_proba")[:, 1]
        result[f"auc_{label}"] = round(float(roc_auc_score(is_val, p)), 3)
    return result


def holdout_figures(train, X, y):
    is_test = (train["date"] >= TEMPORAL_CUTOFF).to_numpy()
    model = final_model().fit(X[FEATURES][~is_test], y[~is_test])
    dist = train["distance"].to_numpy()[is_test]
    truth = np.exp(y[is_test].to_numpy()) * dist
    pred = np.exp(model.predict(X[FEATURES][is_test])) * dist

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax = axes[0]
    ax.scatter(truth, pred, s=6, alpha=0.5, color=ACCENT)
    lim = [0, max(truth.max(), pred.max()) * 1.03]
    ax.plot(lim, lim, color=BAD, linewidth=1)
    ax.set(xlim=lim, ylim=lim, xlabel="actual posted_rate ($)", ylabel="predicted ($)")
    ax.set_title("Temporal holdout (22-28 Jan): predicted vs actual", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax = axes[1]
    ape = 100 * (pred - truth) / truth
    ax.hist(ape, bins=60, color=ACCENT)
    ax.set_xlabel("percentage error (%)")
    ax.set_title(f"Error distribution, MAPE {np.abs(ape).mean():.2f}%", loc="left", fontsize=11, fontweight="bold", color=INK)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "temporal_holdout.png")
    plt.close(fig)

    imp = permutation_importance(model, X[FEATURES][is_test], y[is_test], n_repeats=5, random_state=0,
                                 scoring="neg_mean_squared_error")
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    ax.barh(np.array(FEATURES)[order], imp.importances_mean[order], color=ACCENT)
    ax.set_xlabel("increase in MSE (log rate/mile) when shuffled")
    ax.set_title("Permutation importance (temporal holdout)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png")
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    train = load_training_data()
    X, y = build_features(train), target(train)
    print(f"Clean training rows: {len(train):,}  ({train.date.min().date()} to {train.date.max().date()})")

    print("\nModel comparison")
    table = compare_models(train, X, y)
    table.to_csv(REPORTS / "validation_results.csv", index=False)

    print("\nFeature ablations")
    ablations = feature_ablations(train, X, y)
    print(" ", ablations)

    print("\nLeakage demo (quote_signal)")
    leak = leakage_demo(train, X, y)
    print(" ", leak)

    print("\nAdversarial validation (train vs validation.csv)")
    adv = adversarial_check()
    print(" ", adv)

    holdout_figures(train, X, y)

    model = clone(final_model()).fit(X[FEATURES], y)
    joblib.dump({"model": model, "features": FEATURES}, MODELS / "model.joblib")

    summary = {
        "training_rows": len(train),
        "split_sizes": {
            "temporal_train": int((train.date < TEMPORAL_CUTOFF).sum()),
            "temporal_test": int((train.date >= TEMPORAL_CUTOFF).sum()),
        },
        "results": table.round(4).to_dict(orient="records"),
        "ablations": ablations,
        "leakage_demo": leak,
        "adversarial_validation": adv,
        "final_model": "0.85 x HistGradientBoosting + 0.15 x Ridge on log(rate per mile)",
    }
    (REPORTS / "validation_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved model to {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
