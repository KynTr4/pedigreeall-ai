# Feature Importance

Generated: 2026-06-30 12:43:07

Importance is computed on the 2026 holdout model trained only on internally complete race fields before 2026. Historical current-race HP is forbidden; the contract uses one-race-lagged pre_race_handicap_rating.

## Logistic

| feature | shap_mean_abs | gain_importance | permutation_importance |
| --- | --- | --- | --- |
| race_class | 0.730905 | N/A | 0.017307 |
| surface | 0.568410 | N/A | 0.000008 |
| track | 0.312606 | N/A | 0.000121 |
| last_3_avg_position | 0.311888 | N/A | 0.008332 |
| draw | 0.271693 | N/A | 0.001394 |
| pre_race_handicap_rating | 0.172558 | N/A | 0.003338 |
| last_10_avg_position | 0.153146 | N/A | 0.001829 |
| surface_win_rate | 0.076174 | N/A | 0.000879 |
| days_since_last_race | 0.055287 | N/A | 0.001081 |
| track_win_rate | 0.054216 | N/A | 0.000191 |
| carried_weight | 0.050420 | N/A | 0.000754 |
| surface_change | 0.049119 | N/A | 0.000570 |
| class_change | 0.045384 | N/A | 0.000221 |
| distance | 0.041830 | N/A | -0.000011 |
| jockey_horse_win_rate | 0.031922 | N/A | -0.000002 |
| distance_change | 0.019920 | N/A | 0.000068 |
| last_5_avg_position | 0.019024 | N/A | -0.000229 |
| distance_win_rate | 0.018936 | N/A | 0.000007 |
| trainer_horse_win_rate | 0.011596 | N/A | -0.000093 |
| weight_change | 0.001786 | N/A | -0.000027 |

## Catboost

| feature | shap_mean_abs | gain_importance | permutation_importance |
| --- | --- | --- | --- |
| last_3_avg_position | 0.256978 | 7.985454 | 0.006945 |
| pre_race_handicap_rating | 0.247946 | 22.937914 | 0.020296 |
| draw | 0.174723 | 33.385468 | -0.000541 |
| race_class | 0.155159 | 7.694372 | 0.019071 |
| carried_weight | 0.111682 | 2.364636 | 0.002227 |
| last_5_avg_position | 0.104652 | 4.055414 | 0.002080 |
| last_10_avg_position | 0.088946 | 3.030105 | 0.002543 |
| days_since_last_race | 0.075940 | 2.128022 | 0.001668 |
| track | 0.057693 | 2.621496 | 0.001361 |
| surface_win_rate | 0.054476 | 2.088170 | 0.002052 |
| track_win_rate | 0.052020 | 1.631338 | 0.001715 |
| jockey_horse_win_rate | 0.044541 | 0.837105 | 0.000125 |
| trainer_horse_win_rate | 0.041253 | 4.531854 | 0.002850 |
| weight_change | 0.038008 | 0.916325 | 0.001208 |
| surface | 0.035951 | 1.069444 | 0.001202 |
| surface_change | 0.022995 | 0.508778 | 0.000689 |
| distance | 0.020233 | 0.646717 | 0.000557 |
| distance_change | 0.017017 | 0.414895 | 0.000680 |
| class_change | 0.013816 | 0.602182 | 0.000196 |
| distance_win_rate | 0.010734 | 0.550312 | 0.000058 |

## Xgboost

| feature | shap_mean_abs | gain_importance | permutation_importance |
| --- | --- | --- | --- |
| last_3_avg_position | 0.303675 | 586.924133 | 0.004624 |
| pre_race_handicap_rating | 0.275494 | 275.335541 | 0.003982 |
| draw | 0.245300 | 688.172180 | -0.002198 |
| race_class | 0.201637 | 3120.085884 | 0.014056 |
| carried_weight | 0.116680 | 102.008537 | 0.002219 |
| last_5_avg_position | 0.084658 | 251.181610 | -0.003052 |
| days_since_last_race | 0.071967 | 71.183769 | -0.001074 |
| track | 0.055200 | 556.620342 | 0.000016 |
| surface_win_rate | 0.052885 | 73.308426 | -0.007594 |
| last_10_avg_position | 0.051151 | 128.531433 | -0.002153 |
| trainer_horse_win_rate | 0.049271 | 194.636963 | -0.002746 |
| track_win_rate | 0.040739 | 72.224854 | -0.005488 |
| surface | 0.030423 | 237.171741 | 0.000936 |
| class_change | 0.021936 | 53.153065 | 0.000708 |
| weight_change | 0.019357 | 30.114885 | -0.000999 |
| distance | 0.018217 | 40.210960 | 0.000301 |
| jockey_horse_win_rate | 0.013077 | 39.945347 | -0.003399 |
| distance_win_rate | 0.012870 | 31.569813 | -0.001037 |
| distance_change | 0.012726 | 26.020905 | 0.000068 |
| surface_change | 0.007567 | 35.845760 | 0.000641 |

Logistic regression has no tree gain; its gain cells are intentionally N/A. CatBoost gain is native PredictionValuesChange; XGBoost gain is booster split gain.
