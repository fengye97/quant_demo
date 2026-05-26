# Holdout 报告 — sp500_timing

- 生成时间: 2026-05-25T16:22:06
- 训练 cutoff: 2025-11-30
- Holdout 区间: **2025-12-01 ~ 2026-05-22**  (113 bars)
- Profile 来源: strategy/best_profile_sp500_timing.json

## 调参网格选出的最优参数
| 参数 | 取值 |
|------|------|
| `fast_window` | `15` |
| `slow_window` | `100` |
| `momentum_window` | `80` |

## 训练区评分（来自 walk-forward 选优阶段）
- 评分公式: `0.4*Calmar(6m) + 0.3*Calmar(1y) + 0.3*Calmar(full_pre_cutoff)  [floor: full_nav >= 1.00*default]`
- 综合分: **0.5080**  (maxDD 阈值 0.2)

| 窗口 | Calmar | 年化收益 | 最大回撤 | 平均仓位 | 调仓次数 |
|------|--------|----------|----------|----------|----------|
| recent_6m | 1.120 | 8.28% | -7.41% | 55.28% | 36 |
| recent_1y | -0.120 | -0.97% | -7.76% | 38.43% | 48 |
| full_pre_cutoff | 0.320 | 4.12% | -12.97% | 25.98% | 412 |

## Holdout 区间表现（**只读，未参与选优**）
| 指标 | 取值 |
|------|------|
| 累积净值 | 0.9547 |
| 年化收益 | -9.37% |
| 最大回撤 | -4.93% |
| Calmar | -1.900 |
| 平均仓位 | 17.92% |
| 调仓次数 | 22 |

### 训练区 vs Holdout Calmar 对比
| 窗口 | Calmar |
|------|--------|
| 训练区 recent_6m | 1.120 |
| 训练区 recent_1y | -0.120 |
| 训练区 full_pre_cutoff | 0.320 |
| **Holdout** | **-1.900** |

> 如果 holdout Calmar 显著低于训练区，说明该参数对训练区过拟合；
> 如果接近或更高，说明 walk-forward 选出的参数在 OOS 上稳定。

---

*由 `scripts/build_holdout_reports.py` 自动生成。Holdout 报告只读不参与选优；
如需调整 holdout 起点，请同步修改 `web_app.py` 的 `HOLDOUT_START`。*
