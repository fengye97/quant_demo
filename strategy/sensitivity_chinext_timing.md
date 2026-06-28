# Timing sensitivity audit — chinext_timing

## Best profile
- best params: `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.0}`
- score: **10.7**
- floor ratio: **1.0** × default_full_nav `1.3361`

> 注：`momentum_threshold=0.0` 不能做乘法 ±20% 扰动；本报告改为比较离散邻域值（如 0.0 / 0.01 / 0.02）。

## Local neighborhood rows
- rows in local neighborhood: **24**

| rank | score | discarded | recent_6m | recent_1y | full_pre_cutoff | params |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 10.7 |  | 10.7 | 10.7 | 10.7 | `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.0}` |
| 2 | 9.41 |  | 9.41 | 9.41 | 9.41 | `{'momentum_short_window': 15, 'momentum_long_window': 60, 'trend_window': 40, 'momentum_threshold': 0.02}` |
| 3 | 9.13 |  | 9.13 | 9.13 | 9.13 | `{'momentum_short_window': 15, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.02}` |
| 4 | 8.66 |  | 8.66 | 8.66 | 8.66 | `{'momentum_short_window': 15, 'momentum_long_window': 40, 'trend_window': 40, 'momentum_threshold': 0.02}` |
| 5 | -inf | full_pre_cutoff final_nav=1.1491 < floor=1.3361 (=1.00*default 1.3361) | 2.66 | 2.66 | 2.66 | `{'momentum_short_window': 10, 'momentum_long_window': 40, 'trend_window': 40, 'momentum_threshold': 0.02}` |
| 6 | -inf | full_pre_cutoff final_nav=1.2528 < floor=1.3361 (=1.00*default 1.3361) | 5.59 | 5.59 | 5.59 | `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 40, 'momentum_threshold': 0.01}` |
| 7 | -inf | full_pre_cutoff final_nav=1.1940 < floor=1.3361 (=1.00*default 1.3361) | 3.55 | 3.55 | 3.55 | `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 40, 'momentum_threshold': 0.02}` |
| 8 | -inf | full_pre_cutoff final_nav=1.2672 < floor=1.3361 (=1.00*default 1.3361) | 5.95 | 5.95 | 5.95 | `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.01}` |
| 9 | -inf | full_pre_cutoff final_nav=1.2078 < floor=1.3361 (=1.00*default 1.3361) | 3.83 | 3.83 | 3.83 | `{'momentum_short_window': 10, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.02}` |
| 10 | -inf | full_pre_cutoff final_nav=1.2986 < floor=1.3361 (=1.00*default 1.3361) | 6.26 | 6.26 | 6.26 | `{'momentum_short_window': 15, 'momentum_long_window': 40, 'trend_window': 40, 'momentum_threshold': 0.0}` |
| 11 | -inf | full_pre_cutoff final_nav=1.1550 < floor=1.3361 (=1.00*default 1.3361) | 2.78 | 2.78 | 2.78 | `{'momentum_short_window': 10, 'momentum_long_window': 40, 'trend_window': 60, 'momentum_threshold': 0.02}` |
| 12 | -inf | full_pre_cutoff final_nav=1.2843 < floor=1.3361 (=1.00*default 1.3361) | 8.33 | 8.33 | 8.33 | `{'momentum_short_window': 10, 'momentum_long_window': 40, 'trend_window': 40, 'momentum_threshold': 0.0}` |

## One-parameter stability summary
| parameter | value | best? | grid_points | valid_points | discarded_points | best_score | mean_score | top_recent_6m | top_recent_1y | top_full |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| momentum_short_window | 10.0 | Y | 12 | 1 | 11 | 10.7 | 10.7 | 10.7 | 10.7 | 10.7 |
| momentum_short_window | 15.0 |  | 12 | 3 | 9 | 9.41 | 9.066666666666666 | 9.41 | 9.41 | 9.41 |
| momentum_long_window | 40.0 |  | 12 | 1 | 11 | 8.66 | 8.66 | 8.66 | 8.66 | 8.66 |
| momentum_long_window | 60.0 | Y | 12 | 3 | 9 | 10.7 | 9.746666666666668 | 10.7 | 10.7 | 10.7 |
| trend_window | 40.0 |  | 12 | 2 | 10 | 9.41 | 9.035 | 9.41 | 9.41 | 9.41 |
| trend_window | 60.0 | Y | 12 | 2 | 10 | 10.7 | 9.915 | 10.7 | 10.7 | 10.7 |
| momentum_threshold | 0.0 | Y | 8 | 1 | 7 | 10.7 | 10.7 | 10.7 | 10.7 | 10.7 |
| momentum_threshold | 0.01 |  | 8 | 0 | 8 | nan | nan | 7.26 | 7.26 | 7.26 |
| momentum_threshold | 0.02 |  | 8 | 3 | 5 | 9.41 | 9.066666666666666 | 9.41 | 9.41 | 9.41 |

