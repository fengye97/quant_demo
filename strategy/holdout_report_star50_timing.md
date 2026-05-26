# Holdout 报告 — star50_timing

- 生成时间: 2026-05-25T16:22:06
- 训练 cutoff: 2025-11-30
- Holdout 区间: **2025-12-01 ~ 2026-05-22**  (113 bars)
- Profile 来源: strategy/best_profile_star50_timing.json

## 调参网格选出的最优参数
| 参数 | 取值 |
|------|------|
| `breakout_window` | `8` |
| `exit_window` | `3` |
| `trend_window` | `30` |
| `base_floor` | `0.7` |

## 训练区评分（来自 walk-forward 选优阶段）
- 评分公式: `window_score = (excess_return - 1.5*max(0, dd_excess)) * 100; score = 0.4*ws(6m) + 0.3*ws(1y) + 0.3*ws(full_pre_cutoff) [discard: any window excess_ret < -5.0pp or dd_excess > 5.0pp; full_nav >= 1.00*default]`
- 综合分: **-0.3079**  (maxDD 阈值 0.2)

| 窗口 | Calmar | 年化收益 | 最大回撤 | 平均仓位 | 调仓次数 |
|------|--------|----------|----------|----------|----------|
| recent_6m | 7.610 | 105.94% | -13.91% | 79.05% | 7 |
| recent_1y | 7.610 | 105.94% | -13.91% | 79.05% | 7 |
| full_pre_cutoff | 7.610 | 105.94% | -13.91% | 79.05% | 7 |

## Holdout 区间表现（**只读，未参与选优**）
| 指标 | 取值 |
|------|------|
| 累积净值 | 1.4587 |
| 年化收益 | 122.82% |
| 最大回撤 | -13.93% |
| Calmar | 8.820 |
| 平均仓位 | 82.30% |
| 调仓次数 | 6 |

### 训练区 vs Holdout Calmar 对比
| 窗口 | Calmar |
|------|--------|
| 训练区 recent_6m | 7.610 |
| 训练区 recent_1y | 7.610 |
| 训练区 full_pre_cutoff | 7.610 |
| **Holdout** | **8.820** |

> 如果 holdout Calmar 显著低于训练区，说明该参数对训练区过拟合；
> 如果接近或更高，说明 walk-forward 选出的参数在 OOS 上稳定。

---

*由 `scripts/build_holdout_reports.py` 自动生成。Holdout 报告只读不参与选优；
如需调整 holdout 起点，请同步修改 `web_app.py` 的 `HOLDOUT_START`。*
