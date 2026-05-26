# Holdout 报告 — macro_v32_timing

- 生成时间: 2026-05-25T16:22:06
- 训练 cutoff: 2025-11-30
- Holdout 区间: **2025-12-01 ~ 2026-05-22**  (113 bars)
- Profile 来源: strategy/best_profile_macro_v32_timing.json

## 调参网格选出的最优参数
| 参数 | 取值 |
|------|------|
| `sigmoid_k` | `1.2` |
| `max_leverage` | `1.4` |
| `base_position` | `0.45` |
| `inertia` | `0.05` |
| `crisis_vix` | `40.0` |
| `fed_block_weight` | `0.25` |
| `restrictive_threshold` | `0.4` |
| `pivot_relief` | `0.6` |
| `base_floor` | `0.5` |

> Regime 说明：`fed_block_weight` 控制 Fed 因子块权重；`restrictive_threshold` 表示 Fed 仍偏紧的判定线；`pivot_relief` 表示政策转松时对危机扣分的缓冲强度。

## 训练区评分（来自 walk-forward 选优阶段）
- 评分公式: `window_score = (excess_return - 1.5*max(0, dd_excess)) * 100; score = 0.4*ws(6m) + 0.3*ws(1y) + 0.3*ws(full_pre_cutoff) [discard: any window excess_ret < -5.0pp or dd_excess > 5.0pp; full_nav >= 1.00*default]`
- 综合分: **nan**  (maxDD 阈值 0.2)

| 窗口 | Calmar | 年化收益 | 最大回撤 | 平均仓位 | 调仓次数 |
|------|--------|----------|----------|----------|----------|
| recent_6m | 6.660 | 43.89% | -6.59% | 91.48% | 1 |
| recent_1y | 0.920 | 18.39% | -19.93% | 86.68% | 4 |
| full_pre_cutoff | 0.730 | 14.55% | -19.92% | 65.69% | 56 |

## Holdout 区间表现（**只读，未参与选优**）
| 指标 | 取值 |
|------|------|
| 累积净值 | 1.0727 |
| 年化收益 | 16.06% |
| 最大回撤 | -11.86% |
| Calmar | 1.350 |
| 平均仓位 | 80.94% |
| 调仓次数 | 4 |

### 训练区 vs Holdout Calmar 对比
| 窗口 | Calmar |
|------|--------|
| 训练区 recent_6m | 6.660 |
| 训练区 recent_1y | 0.920 |
| 训练区 full_pre_cutoff | 0.730 |
| **Holdout** | **1.350** |

> 如果 holdout Calmar 显著低于训练区，说明该参数对训练区过拟合；
> 如果接近或更高，说明 walk-forward 选出的参数在 OOS 上稳定。

---

*由 `scripts/build_holdout_reports.py` 自动生成。Holdout 报告只读不参与选优；
如需调整 holdout 起点，请同步修改 `web_app.py` 的 `HOLDOUT_START`。*
