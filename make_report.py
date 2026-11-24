"""Build reports/Freight_Rate_Report.docx from the pipeline outputs.

Run:  python make_report.py      (after eda.py, train.py, predict.py, score.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TEAL = RGBColor(0x06, 0x4A, 0x56)


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Inches(w)
    doc.add_paragraph()
    return t


def figure(doc, path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(9)


def main():
    eda = json.loads((REPORTS / "eda_summary.json").read_text())
    res = json.loads((REPORTS / "validation_results.json").read_text())
    results = pd.DataFrame(res["results"])
    pivot = results.pivot(index="model", columns="scheme", values="mape_pct")
    mae = results.pivot(index="model", columns="scheme", values="mae_usd")
    leak, adv = res["leakage_demo"], res["adversarial_validation"]
    q_train, q_val = eda["quality"]
    qs_t, qs_v = eda["quote_signal_train"], eda["quote_signal_validation"]
    dec = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    best = "blend_85gb_15ridge"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    for level in (1, 2):
        doc.styles[f"Heading {level}"].font.color.rgb = TEAL

    title = doc.add_heading("Freight Rate Prediction: Validation & Modelling Report", 0)
    title.runs[0].font.color.rgb = TEAL
    title.runs[0].font.size = Pt(22)

    doc.add_heading("Summary", 1)
    doc.add_paragraph(
        f"I trained on {res['training_rows']:,} clean January 2025 loads to predict posted_rate for 12,000 "
        f"Nov–Dec 2025 loads. The shipped model blends gradient boosting with ridge regression on log(rate per mile). "
        f"It scores {pivot.loc[best, 'temporal']:.2f}% MAPE on a forward-in-time holdout and "
        f"{pivot.loc[best, 'new_city']:.2f}% on lanes with cities it has never seen. "
        f"A per-equipment median baseline scores {pivot.loc['baseline_equipment_median', 'temporal']:.2f}%."
    )
    doc.add_paragraph(
        "The most important decision was excluding quote_signal. It mirrors the label almost exactly in the training "
        "file but carries no information in validation.csv. A model that used it would look excellent in "
        f"cross-validation ({leak['with_quote_signal_as_in_train_mape_pct']:.1f}% MAPE) and then fail on the real "
        f"predictions ({leak['with_quote_signal_as_in_validation_mape_pct']:.1f}% MAPE, worse than the baseline)."
    )

    doc.add_heading("1. Data and key findings", 1)
    for text in [
        f"Labeled data runs from {q_train['date_min']} to {q_train['date_max']}. The loads to predict run from "
        f"{q_val['date_min']} to {q_val['date_max']}. That is a ten-month gap, so the date itself cannot be used as "
        "a feature, and validation has to respect time order.",
        "With label noise removed, posted_rate is almost exactly quote_signal × distance "
        f"(ratio sd {eda['label_ratio_clean_std']:.3f}). Rate per mile falls steeply with distance because fixed costs "
        "are spread over fewer miles on short hauls. It rises with weight and market_index. Median $/mile: "
        + ", ".join(f"{k} {v:.2f}" for k, v in eda["clean_rate_per_mile_by_equipment"].items()) + ".",
        "market_index is a daily market level with a clear weekly cycle: weekend lows, Thursday highs. Nov–Dec has "
        "the same range as January, so the model is not extrapolating. Day of week adds nothing once market_index "
        "is in the model.",
        f"{len(eda['new_cities_in_validation'])} cities in validation never appear in training "
        f"({', '.join(eda['new_cities_in_validation'])}), affecting "
        f"{eda['validation_rows_touching_new_city']:,} rows (12%).",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    figure(doc, FIG / "quote_signal_shift.png",
           "Figure 1. quote_signal depends on distance and equipment in train (left) but is flat noise in validation (right).")

    doc.add_heading("2. Data-quality issues and fixes", 1)
    table(doc, ["Issue", "Train", "Validation", "Fix"], [
        ["quote_signal decoupled from the load in validation",
         f"corr(1/dist) {qs_t['corr_with_inverse_distance']}", f"corr(1/dist) {qs_v['corr_with_inverse_distance']}",
         "Excluded as a feature; used only to audit training labels"],
        ["Corrupted labels (0.2–0.4× or 2.3–5× the expected rate)", q_train["corrupted_label"], "–",
         "Dropped from training and evaluation"],
        ["Missing label", q_train["missing_label"], "–", "Dropped"],
        ["Negative weight (sign flips)", q_train["negative_weight"], q_val["negative_weight"], "abs(weight)"],
        ["Missing weight", q_train["missing_weight"], q_val["missing_weight"],
         "Per-equipment training median + missing flag"],
        ["Missing market_index", q_train["missing_market_index"], q_val["missing_market_index"],
         "Same-day median (it is a daily level)"],
        ["Weight capped at 47,500 lb", q_train["weight_at_cap_47500"], q_val["weight_at_cap_47500"],
         "Kept; valid legal maximum"],
        ["Distance floored at 70 mi", q_train["distance_at_floor_70"], q_val["distance_at_floor_70"],
         "Kept; labels are consistent with it. Circuity feature flags it"],
        ["Unseen cities", "–", eda["validation_rows_touching_new_city"],
         "No city IDs; location via lat/lon so it generalises"],
    ], widths=[2.2, 0.9, 1.0, 2.4])
    figure(doc, FIG / "label_audit.png",
           "Figure 2. Label audit. Clean labels sit within ±1% of quote_signal × distance; 70 rows fall well outside ±10%.",
           width=5.5)

    doc.add_heading("3. Train / test split and validation approach", 1)
    doc.add_paragraph(
        "A single random split would answer the wrong question. The real task predicts loads ten months after "
        "the training period, and 12% of those loads involve unfamiliar cities. So every candidate model was "
        "scored on three schemes, each built to mirror one deployment risk:"
    )
    sizes = res["split_sizes"]
    table(doc, ["Scheme", "How the split is made", "Train / test rows", "Risk it measures"], [
        ["Temporal (primary)", "Train 1–21 Jan, test 22–28 Jan. No future data in training.",
         f"{sizes['temporal_train']:,} / {sizes['temporal_test']:,}", "Drift over time"],
        ["New-city (5 folds)", "Cities shuffled into 5 groups. Rows touching a held-out group are test; rows touching "
         "none are train.", "~2,750 / ~1,550 per fold", "Unseen origins/destinations"],
        ["Random 5-fold", "Shuffled K-fold", "3,444 / 861 per fold", "Variance / stability"],
    ], widths=[1.3, 2.7, 1.2, 1.3])
    for text in [
        "Corrupted labels are removed before splitting. Test metrics are therefore measured against trustworthy "
        "targets, and a handful of 5× typos cannot dominate the error.",
        "Metrics are computed on posted_rate in dollars: MAPE (primary), MAE, RMSE and share within ±5%. "
        "Model selection and every hyperparameter choice used the temporal and new-city scores, never the random one.",
        f"Leakage test: the same model fitted with quote_signal scored "
        f"{leak['with_quote_signal_as_in_train_mape_pct']:.2f}% on the temporal holdout. Shuffling quote_signal in the "
        f"holdout, to mimic validation.csv, raised that to {leak['with_quote_signal_as_in_validation_mape_pct']:.2f}%. "
        f"Without the feature the model scores {leak['without_quote_signal_mape_pct']:.2f}% and is unaffected.",
        f"Adversarial validation (classifier separating train vs validation rows): AUC "
        f"{adv['auc_model_features']:.3f} on the model's features versus "
        f"{adv['auc_model_features_plus_quote_signal']:.3f} with quote_signal added. The small remaining shift comes "
        "from the new cities and a slightly higher market_index level (AUC ≈ 0.50 when both are removed). "
        "The inputs the model relies on are stable between the two files.",
        "The final model is refit on all clean January rows before predicting validation.csv.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("4. Model and results", 1)
    doc.add_paragraph(
        "Target: log(posted_rate / distance). Predicting per-mile cost removes the dominant linear distance effect, "
        "and the log turns multiplicative pricing effects into additive ones. Errors become relative, which suits "
        "loads ranging from $150 to $6,000. Features: log distance, 1/distance, weight, equipment, market_index, "
        "pickup/delivery lat-lon, lane direction (Δlat, Δlon) and road circuity."
    )
    names = {"baseline_equipment_median": "Equipment median (baseline)", "ridge": "Ridge regression",
             "gradient_boosting": "Gradient boosting", best: "Blend 0.85 GB + 0.15 Ridge (shipped)"}
    table(doc, ["Model", "Temporal MAPE", "New-city MAPE", "Random MAPE", "Temporal MAE"], [
        [names[m], f"{pivot.loc[m, 'temporal']:.2f}%", f"{pivot.loc[m, 'new_city']:.2f}%",
         f"{pivot.loc[m, 'random_kfold']:.2f}%", f"${mae.loc[m, 'temporal']:.0f}"]
        for m in names
    ], widths=[2.3, 1.1, 1.1, 1.1, 1.0])
    doc.add_paragraph(
        "Why this model: gradient boosting captures the non-linear distance curve and the equipment × distance and "
        "regional interactions. A light ridge component adds smooth behaviour where trees extrapolate poorly, such as "
        "new cities and edge market levels, and it improved every scheme. Shallow trees (7 leaves) beat deeper ones. "
        "Out-of-fold residuals show no remaining structure by city, distance, weight or equipment: per-city residual "
        "spread (0.0024) equals pure sampling noise (0.0025). The model is close to the noise floor for these inputs."
    )
    figure(doc, FIG / "temporal_holdout.png", "Figure 3. Temporal holdout: predicted vs actual and error distribution.")
    figure(doc, FIG / "feature_importance.png", "Figure 4. Permutation importance on the temporal holdout.", width=4.8)

    doc.add_heading("5. Fixed December prediction chart", 1)
    doc.add_paragraph(
        "The December rows include lane, distance, equipment, weight and date only. Pickup and delivery coordinates "
        "come from the city lookup. market_index, the only date-driven input, is that day's median across the "
        "validation loads. Those loads cover every December day; only their features are used, never labels. "
        f"Predictions range from ${dec.predicted_rate.min():,.2f} to ${dec.predicted_rate.max():,.2f} and follow "
        "the weekly market cycle. As a sanity check, January Lexington → Fort Wayne Dry Van loads (~360 mi) "
        "went for $758–$808."
    )
    figure(doc, ROOT / "scorer_results" / "candidate_december.png",
           "Figure 5. Output of the provided score.py (scorer_results/candidate_december.png).")

    doc.add_heading("6. Limitations and next steps", 1)
    for text in [
        "Only four weeks of labels. Seasonal effects (holiday peaks, year-end capacity) can only reach the model "
        "through market_index. A few labeled Nov–Dec loads would be the most valuable addition.",
        "Market level comes from a single index. Lane-level spot data or tender rejection rates would add the "
        "regional demand signal that coordinates can only approximate.",
        "If quote_signal is genuinely available at quote time in production, its pipeline should be investigated. "
        "In this validation file it is not usable.",
        "Prediction intervals, e.g. quantile boosting, would help a pricing desk know when to trust the number.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("Reproduce", 2)
    doc.add_paragraph("pip install -r requirements.txt\npython eda.py && python train.py && python predict.py\n"
                      "python score.py --predictions validation_predictions.csv "
                      "--december-predictions data/december_chart_inputs.csv")

    out = REPORTS / "Freight_Rate_Report.docx"
    doc.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
