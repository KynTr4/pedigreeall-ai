# ROI Simulation Report — NOT CERTIFIED

Generated: 2026-06-30 12:43:07

Historical odds have no immutable pre-race timestamp. ROI is diagnostic only and must not be used as a live-return claim.

Each selected horse receives 1 unit. Decimal `odds` is treated as total return including stake. No commission, limit, slippage, dead heat, or late odds movement is modeled.

| split | model | strategy | total_bets | winning_bets | profit | roi | average_odds | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | catboost | top_1 | 1642.0 | 452 | -514.0500 | -0.3131 | 4.2475 | 522.7500 |
| holdout | catboost | top_2 | 3284.0 | 771 | -1070.9000 | -0.3261 | 4.8600 | 1079.1500 |
| holdout | catboost | top_3 | 4925.0 | 1012 | -1702.7000 | -0.3457 | 5.6223 | 1710.1000 |
| holdout | ensemble | top_1 | 1642.0 | 453 | -475.2500 | -0.2894 | 4.3775 | 484.3500 |
| holdout | ensemble | top_2 | 3284.0 | 774 | -1022.5500 | -0.3114 | 5.0200 | 1031.2000 |
| holdout | ensemble | top_3 | 4925.0 | 999 | -1718.2000 | -0.3489 | 5.8584 | 1722.4500 |
| holdout | logistic | top_1 | 1642.0 | 438 | -463.1000 | -0.2820 | 4.7952 | 476.7500 |
| holdout | logistic | top_2 | 3284.0 | 736 | -1028.4500 | -0.3132 | 5.6918 | 1037.1000 |
| holdout | logistic | top_3 | 4925.0 | 961 | -1755.8500 | -0.3565 | 6.5947 | 1763.2500 |
| holdout | xgboost | top_1 | 1642.0 | 439 | -515.6000 | -0.3140 | 4.7618 | 530.5000 |
| holdout | xgboost | top_2 | 3284.0 | 754 | -1146.4000 | -0.3491 | 5.2172 | 1154.4500 |
| holdout | xgboost | top_3 | 4925.0 | 977 | -1832.5000 | -0.3721 | 6.0544 | 1836.7500 |
| test | catboost | top_1 | 2867.0 | 740 | -733.2500 | -0.2558 | 5.0809 | 741.2500 |
| test | catboost | top_2 | 5734.0 | 1338 | -1383.2500 | -0.2412 | 5.8490 | 1402.7000 |
| test | catboost | top_3 | 8579.0 | 1730 | -2319.2000 | -0.2703 | 6.7191 | 2320.6500 |
| test | ensemble | top_1 | 2867.0 | 745 | -676.2500 | -0.2359 | 5.1096 | 684.4000 |
| test | ensemble | top_2 | 5734.0 | 1326 | -1421.2500 | -0.2479 | 5.9318 | 1432.0000 |
| test | ensemble | top_3 | 8579.0 | 1727 | -2366.1000 | -0.2758 | 6.8392 | 2366.4500 |
| test | logistic | top_1 | 2867.0 | 719 | -702.1500 | -0.2449 | 5.5792 | 708.5000 |
| test | logistic | top_2 | 5734.0 | 1252 | -1606.4500 | -0.2802 | 6.7486 | 1619.0500 |
| test | logistic | top_3 | 8579.0 | 1656 | -2441.4000 | -0.2846 | 7.7717 | 2441.7500 |
| test | xgboost | top_1 | 2867.0 | 752 | -557.1500 | -0.1943 | 5.6802 | 572.1000 |
| test | xgboost | top_2 | 5734.0 | 1287 | -1467.6000 | -0.2559 | 6.1891 | 1470.3000 |
| test | xgboost | top_3 | 8579.0 | 1713 | -2296.3500 | -0.2677 | 7.0294 | 2297.8000 |
| validation | catboost | top_1 | 799.0 | 212 | -277.9000 | -0.3478 | 5.6265 | 293.4000 |
| validation | catboost | top_2 | 1598.0 | 391 | -403.1000 | -0.2523 | 5.9715 | 419.1000 |
| validation | catboost | top_3 | 2363.0 | 520 | -630.9000 | -0.2670 | 6.7543 | 666.2000 |
| validation | ensemble | top_1 | 799.0 | 216 | -285.0500 | -0.3568 | 5.4500 | 290.3500 |
| validation | ensemble | top_2 | 1598.0 | 388 | -406.1500 | -0.2542 | 5.9740 | 430.3000 |
| validation | ensemble | top_3 | 2363.0 | 528 | -577.9500 | -0.2446 | 6.7617 | 652.4000 |
| validation | logistic | top_1 | 799.0 | 207 | -262.5000 | -0.3285 | 5.4199 | 263.6500 |
| validation | logistic | top_2 | 1598.0 | 379 | -393.5000 | -0.2462 | 6.3335 | 441.0500 |
| validation | logistic | top_3 | 2363.0 | 506 | -616.2000 | -0.2608 | 7.2295 | 673.3500 |
| validation | xgboost | top_1 | 799.0 | 206 | -262.2000 | -0.3282 | 6.5483 | 279.0000 |
| validation | xgboost | top_2 | 1598.0 | 384 | -380.6500 | -0.2382 | 6.6439 | 404.7500 |
| validation | xgboost | top_3 | 2363.0 | 514 | -603.2500 | -0.2553 | 7.2383 | 666.4500 |

## AGF Value Bet

Not calculated. `agf` has zero populated rows and `agf_percent`/`agf_rank` contain only `not_found`; fabricating an AGF comparison would invalidate the test. The CSV records this strategy as `unavailable_missing_agf`.
