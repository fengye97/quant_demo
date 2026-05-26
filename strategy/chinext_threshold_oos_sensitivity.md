# Chinext momentum_threshold OOS sensitivity — 2026-05-23T22:18:50

## Context
- training best params: `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.0}`
- current best threshold: **0.0**
- training floor: **1.00 × default_full_nav = 1.3361**
- holdout window: **2025-12-01 ~ 2026-05-21**

> 说明：由于 best 值是 `0.0`，这里不用“乘法 ±20%”，而改成绝对扰动 `0.000 / 0.005 / 0.010 / 0.015 / 0.020`。
> holdout 指标只读展示，不参与选优，不回写 `best_profile`。

## Training + holdout comparison
| threshold | train_score | train_discarded | train_6m_calmar | train_1y_calmar | train_full_calmar | holdout_calmar | holdout_annret | holdout_maxdd | holdout_final_nav | holdout_rows |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 10.7 |  | 10.7 | 10.7 | 10.7 | 2.16 | 0.1058 | -0.0489 | 1.0482 | 112 |
| 0.005 | -inf | full_pre_cutoff final_nav=1.3084 < floor=1.3361 (=1.00*default 1.3361) | 9.13 | 9.13 | 9.13 | 2.43 | 0.1193 | -0.049 | 1.0542 | 112 |
| 0.010 | -inf | full_pre_cutoff final_nav=1.2672 < floor=1.3361 (=1.00*default 1.3361) | 5.95 | 5.95 | 5.95 | 2.76 | 0.1352 | -0.049 | 1.0612 | 112 |
| 0.015 | -inf | full_pre_cutoff final_nav=1.2779 < floor=1.3361 (=1.00*default 1.3361) | 6.22 | 6.22 | 6.22 | 0.05 | 0.0024 | -0.049 | 1.0011 | 112 |
| 0.020 | -inf | full_pre_cutoff final_nav=1.2078 < floor=1.3361 (=1.00*default 1.3361) | 3.83 | 3.83 | 3.83 | 0.66 | 0.0321 | -0.0489 | 1.0149 | 112 |

## Takeaways
- 在这组最小扰动里，holdout Calmar 最优的是 `0.010`，而训练 best 是 `0.000`。
- 训练 best `0.000` 的 holdout: `Calmar=2.16`, `annRet=0.1058`, `maxDD=-0.0489`, `final_nav=1.0482`。
- 若某个阈值在训练区已被 `full_nav_floor` 或 `maxDD` 约束淘汰，即使 holdout 看起来更好，也**不能**据此回写为 best。

