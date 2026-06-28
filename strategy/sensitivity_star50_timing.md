# Timing sensitivity audit — star50_timing

## Best profile
- best params: `{'breakout_window': 8, 'exit_window': 3, 'trend_window': 60}`
- score: **10.67**
- floor ratio: **1.0** × default_full_nav `1.1353`

## Local neighborhood rows
- rows in local neighborhood: **8**

| rank | score | discarded | recent_6m | recent_1y | full_pre_cutoff | params |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 10.67 | nan | 10.67 | 10.67 | 10.67 | `{'breakout_window': 8.0, 'exit_window': 3.0, 'trend_window': 60.0}` |
| 2 | 10.67 | nan | 10.67 | 10.67 | 10.67 | `{'breakout_window': 10.0, 'exit_window': 3.0, 'trend_window': 60.0}` |
| 3 | 10.15 | nan | 10.15 | 10.15 | 10.15 | `{'breakout_window': 10.0, 'exit_window': 3.0, 'trend_window': 40.0}` |
| 4 | 10.15 | nan | 10.15 | 10.15 | 10.15 | `{'breakout_window': 8.0, 'exit_window': 3.0, 'trend_window': 40.0}` |
| 5 | 4.63 | nan | 4.63 | 4.63 | 4.63 | `{'breakout_window': 8.0, 'exit_window': 5.0, 'trend_window': 60.0}` |
| 6 | 4.63 | nan | 4.63 | 4.63 | 4.63 | `{'breakout_window': 10.0, 'exit_window': 5.0, 'trend_window': 60.0}` |
| 7 | 4.34 | nan | 4.34 | 4.34 | 4.34 | `{'breakout_window': 8.0, 'exit_window': 5.0, 'trend_window': 40.0}` |
| 8 | 4.34 | nan | 4.34 | 4.34 | 4.34 | `{'breakout_window': 10.0, 'exit_window': 5.0, 'trend_window': 40.0}` |

## One-parameter stability summary
| parameter | value | best? | grid_points | valid_points | discarded_points | best_score | mean_score | top_recent_6m | top_recent_1y | top_full |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| breakout_window | 8 | Y | 4 | 4 | 0 | 10.67 | 7.4475 | 10.67 | 10.67 | 10.67 |
| breakout_window | 10 |  | 4 | 4 | 0 | 10.67 | 7.4475 | 10.67 | 10.67 | 10.67 |
| exit_window | 3 | Y | 4 | 4 | 0 | 10.67 | 10.41 | 10.67 | 10.67 | 10.67 |
| exit_window | 5 |  | 4 | 4 | 0 | 4.63 | 4.484999999999999 | 4.63 | 4.63 | 4.63 |
| trend_window | 40 |  | 4 | 4 | 0 | 10.15 | 7.245 | 10.15 | 10.15 | 10.15 |
| trend_window | 60 | Y | 4 | 4 | 0 | 10.67 | 7.6499999999999995 | 10.67 | 10.67 | 10.67 |

