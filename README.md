# Freight Rate Prediction

Predicts `posted_rate` for 12,000 truckload shipments (Nov–Dec 2025) from a labeled set of 4,375 January 2025 loads.

**Result:** 1.51% MAPE on a forward-in-time holdout and 1.60% on lanes with completely unseen cities. A naive per-equipment baseline gets 8.0%.

The full write-up is in [`reports/Freight_Rate_Report.docx`](reports/Freight_Rate_Report.docx).

## Quick start

```bash
python -m pip install -r requirements.txt

python eda.py          # data-quality checks + figures      -> reports/
python train.py        # validation experiments + final fit  -> models/model.joblib
python predict.py      # validation_predictions.csv + fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_report.py  # rebuilds reports/Freight_Rate_Report.docx
```

Tested with Python 3.12, pandas 3.0, scikit-learn 1.9. The whole pipeline runs in about two minutes on a laptop.

## What I found and what I did about it

| Finding | Decision |
|---|---|
| `quote_signal` tracks the label almost exactly in train (`rate ≈ quote_signal × distance`, ±1%), but in `validation.csv` it is unrelated to distance or equipment (corr with 1/distance: 0.77 in train vs −0.003 in validation). | **Excluded as a feature.** With it, holdout error looks like 0.9%. Once it behaves the way it does in validation, error jumps to 11.9%, which is worse than the baseline. It is still used inside train to audit labels. |
| 70 labels (1.6%) are off by a 0.2–0.4× or 2.3–5× factor relative to `quote_signal × distance`; 1 row has no label. | Dropped from training and evaluation. |
| Negative weights (29 train / 145 validation) have the same magnitude distribution as positive ones. | Treated as sign flips: `abs()`. |
| Missing weight / `market_index`. | Weight: per-equipment training median plus a missing flag. `market_index`: that day's median (it is a daily level with ~0.025 row noise). |
| 8 cities in validation never appear in train (1,447 rows). | No city IDs. Location goes in as lat/lon, which works for unseen cities. |
| Labels cover 1–28 Jan; predictions are for Nov–Dec. | Primary validation is a time-ordered holdout. The calendar only enters through `market_index`. |

## Model

The target is `log(posted_rate / distance)`, so the model learns what each mile costs and errors are measured in relative terms. The final model is **0.85 × HistGradientBoosting + 0.15 × Ridge**, using 14 features: distance (log and 1/x), weight, equipment, `market_index`, pickup and delivery coordinates, lane direction, and road circuity.

## Validation

| Scheme | What it simulates | Train / test rows |
|---|---|---|
| **Temporal** (primary) | Predicting the future: train 1–21 Jan, test 22–28 Jan | 3,281 / 1,024 |
| **New-city** (5 folds) | Lanes touching cities never seen in training | ~2,750 / ~1,550 per fold |
| Random 5-fold | Stability check | 3,444 / 861 per fold |

| Model | Temporal MAPE | New-city MAPE | Random MAPE |
|---|---|---|---|
| Equipment median (baseline) | 8.01% | 8.12% | 8.07% |
| Ridge | 1.93% | 1.97% | 1.90% |
| Gradient boosting | 1.53% | 1.63% | 1.53% |
| **Blend (shipped)** | **1.51%** | **1.60%** | **1.51%** |

## December chart

The December rows only include lane, distance, equipment, weight and date. Coordinates come from the city lookup. `market_index`, the one date-driven input, is that day's median across the validation loads, which cover all of December. The resulting curve ($781–$798) follows the weekly market cycle. Real Lexington → Fort Wayne Dry Van loads in January went for $758–$808.

## Layout

```
data/                     input files (+ december_chart_inputs.csv filled by predict.py)
src/freight/data.py       loading, label audit, cleaning rules
src/freight/features.py   feature engineering
src/freight/models.py     baseline, ridge, boosting, blend
src/freight/validation.py split schemes + metrics
eda.py / train.py / predict.py / make_report.py
reports/                  figures, validation_results.json, report
validation_predictions.csv
scorer_results/candidate_december.png
```

The original assessment instructions are in `Freight_Rate_ML_Assessment.pdf` and `reports/original_readme.md`.
