# Timing sensitivity audit — csi1000_timing

## Best profile
- best params: `{'breakout_window': 10, 'exit_window': 5, 'trend_window': 50}`
- score: **4.917**
- floor ratio: **1.0** × default_full_nav `1.0504`

## Local neighborhood rows
- rows in local neighborhood: **12**

| rank | score | discarded | recent_6m | recent_1y | full_pre_cutoff | params |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 4.917 | nan | 9.69 | 2.98 | 0.49 | `{'breakout_window': 10, 'exit_window': 5, 'trend_window': 50}` |
| 2 | 4.842 | nan | 9.69 | 2.98 | 0.24 | `{'breakout_window': 15, 'exit_window': 5, 'trend_window': 50}` |
| 3 | 4.760999999999999 | nan | 9.69 | 2.52 | 0.43 | `{'breakout_window': 10, 'exit_window': 7, 'trend_window': 50}` |
| 4 | 4.689 | nan | 9.69 | 2.52 | 0.19 | `{'breakout_window': 15, 'exit_window': 7, 'trend_window': 50}` |
| 5 | 3.5860000000000003 | nan | 6.43 | 2.83 | 0.55 | `{'breakout_window': 10, 'exit_window': 5, 'trend_window': 30}` |
| 6 | 3.526 | nan | 6.43 | 2.69 | 0.49 | `{'breakout_window': 10, 'exit_window': 7, 'trend_window': 30}` |
| 7 | 3.511 | nan | 6.43 | 2.83 | 0.3 | `{'breakout_window': 15, 'exit_window': 5, 'trend_window': 30}` |
| 8 | 3.454 | nan | 6.43 | 2.69 | 0.25 | `{'breakout_window': 15, 'exit_window': 7, 'trend_window': 30}` |
| 9 | -inf | full_pre_cutoff final_nav=0.9939 < floor=1.0504 (=1.00*default 1.0504) | 10.25 | 3.21 | -0.02 | `{'breakout_window': 15, 'exit_window': 7, 'trend_window': 80}` |
| 10 | -inf | full_pre_cutoff final_nav=1.0067 < floor=1.0504 (=1.00*default 1.0504) | 10.25 | 3.87 | 0.03 | `{'breakout_window': 15, 'exit_window': 5, 'trend_window': 80}` |
| 11 | -inf | full_pre_cutoff final_nav=0.9939 < floor=1.0504 (=1.00*default 1.0504) | 10.25 | 3.21 | -0.02 | `{'breakout_window': 10, 'exit_window': 7, 'trend_window': 80}` |
| 12 | -inf | full_pre_cutoff final_nav=1.0067 < floor=1.0504 (=1.00*default 1.0504) | 10.25 | 3.87 | 0.03 | `{'breakout_window': 10, 'exit_window': 5, 'trend_window': 80}` |

## One-parameter stability summary
| parameter | value | best? | grid_points | valid_points | discarded_points | best_score | mean_score | top_recent_6m | top_recent_1y | top_full |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| breakout_window | 10 | Y | 6 | 4 | 2 | 4.917 | 4.1975 | 9.69 | 2.98 | 0.49 |
| breakout_window | 15 |  | 6 | 4 | 2 | 4.842 | 4.124 | 9.69 | 2.98 | 0.24 |
| exit_window | 5 | Y | 6 | 4 | 2 | 4.917 | 4.214 | 9.69 | 2.98 | 0.49 |
| exit_window | 7 |  | 6 | 4 | 2 | 4.760999999999999 | 4.1075 | 9.69 | 2.52 | 0.43 |
| trend_window | 30 |  | 4 | 4 | 0 | 3.5860000000000003 | 3.5192500000000004 | 6.43 | 2.83 | 0.55 |
| trend_window | 50 | Y | 4 | 4 | 0 | 4.917 | 4.80225 | 9.69 | 2.98 | 0.49 |
| trend_window | 80 |  | 4 | 0 | 4 | nan | nan | 10.25 | 3.21 | -0.02 |

