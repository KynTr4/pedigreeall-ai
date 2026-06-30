# Production Model Training Report v2

Generated: 2026-06-30T09:50:08.530425+00:00

- Historical current-race `GET:Tjk/Get.HP` is forbidden.
- `pre_race_handicap_rating` is one-race-lagged in historical training and direct pre-race program HANDICAP in live scoring.
- Only internally complete race fields are admitted.
- Date parser: `pd.to_datetime(..., dayfirst=True, errors='coerce')`.

- Train: 214,189 rows / 26,458 races (1970-01-08–2025-12-31)
- Holdout: 15,322 rows / 1,642 races (2026-01-01–2026-06-26)

| Model | Top-1 | LogLoss | Brier | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| logistic | 26.6748% | 0.3102 | 0.0890 | 0.4444 | 0.0024 | 0.0048 |
| catboost | 27.5274% | 0.3071 | 0.0886 | 0.3333 | 0.0018 | 0.0036 |
| xgboost | 26.7357% | 0.3136 | 0.0902 | 1.0000 | 0.0006 | 0.0012 |
| ensemble | 27.5883% | 0.3087 | 0.0889 | 0.6667 | 0.0012 | 0.0024 |
