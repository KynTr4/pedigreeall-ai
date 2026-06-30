# Calibration Report

Generated: 2026-06-30 12:43:07

Probabilities are normalized independently inside every race before calibration measurement. ECE uses 10 fixed-width bins.

| split | model | calibration_error | brier_score | log_loss |
| --- | --- | --- | --- | --- |
| validation | logistic | 0.0074 | 0.1095 | 0.3674 |
| validation | catboost | 0.0088 | 0.1097 | 0.3664 |
| validation | xgboost | 0.0126 | 0.1111 | 0.3724 |
| validation | ensemble | 0.0112 | 0.1095 | 0.3666 |
| test | logistic | 0.0062 | 0.0899 | 0.3122 |
| test | catboost | 0.0110 | 0.0893 | 0.3084 |
| test | xgboost | 0.0191 | 0.0909 | 0.3156 |
| test | ensemble | 0.0162 | 0.0896 | 0.3104 |
| holdout | logistic | 0.0078 | 0.0890 | 0.3102 |
| holdout | catboost | 0.0094 | 0.0886 | 0.3071 |
| holdout | xgboost | 0.0161 | 0.0902 | 0.3136 |
| holdout | ensemble | 0.0122 | 0.0889 | 0.3087 |
| all_evaluation_folds | logistic | 0.0054 | 0.0920 | 0.3183 |
| all_evaluation_folds | catboost | 0.0094 | 0.0916 | 0.3151 |
| all_evaluation_folds | xgboost | 0.0172 | 0.0932 | 0.3220 |
| all_evaluation_folds | ensemble | 0.0141 | 0.0918 | 0.3167 |

Curve: `reports/calibration_curve_v2.png`; reliability data: `output/calibration_table_v2.csv`. Empty high-probability bins are retained with count zero.
