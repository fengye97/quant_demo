"""
回测模块 — RL 策略的样本外回测与性能分析

功能:
    1. 加载训练好的 PPO 模型进行样本外推演
    2. 计算关键绩效指标: 年化收益、夏普比率、最大回撤、Calmar 比率、胜率
    3. 生成资金曲线、回撤曲线
    4. 与基准 (等权全市场) 对比

绩效指标说明:
    - 累积净值 (Cumulative Net Value): 初始 1 元的终值
    - 年化收益率 (Annualized Return): (终值)^(252/天数) - 1
    - 年化波动率 (Annualized Volatility): std(日收益) * sqrt(252)
    - 夏普比率 (Sharpe Ratio): (年化收益 - 无风险利率) / 年化波动率
    - 最大回撤 (Max Drawdown): 资金曲线从峰值到谷底的最大跌幅
    - Calmar 比率 (Calmar Ratio): 年化收益 / |最大回撤|
    - 胜率 (Win Rate): 正收益月份占比
    - 换手率 (Turnover): 平均月度单边换手率
    - 信息比率 (Information Ratio): 超额收益 / 跟踪误差

使用方式:
    from rl_stock_selector.backtest import run_backtest, BacktestResult

    result = run_backtest(model, env, df)
    print(result.summary())
    result.plot()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class BacktestResult:
    """回测结果数据结构"""

    cumulative_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    monthly_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark_cumulative: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_series: np.ndarray = field(default_factory=lambda: np.array([]))
    dates: List[str] = field(default_factory=list)
    positions_history: List[Dict] = field(default_factory=list)

    # 计算指标
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    avg_turnover: float = 0.0
    information_ratio: float = 0.0
    n_months: int = 0

    def summary(self) -> str:
        """生成回测摘要"""
        lines = [
            "=" * 60,
            "  RL Stock Selector — 回测评估报告",
            "=" * 60,
            f"  回测周期: {self.dates[0]} ~ {self.dates[-1]} ({self.n_months} 个月)",
            "",
            "  绩效指标:",
            f"    累积净值:       {self.total_return:.4f}",
            f"    年化收益率:     {self.annual_return*100:.2f}%",
            f"    年化波动率:     {self.annual_volatility*100:.2f}%",
            f"    夏普比率:       {self.sharpe_ratio:.4f}",
            f"    最大回撤:       {self.max_drawdown*100:.2f}%",
            f"    Calmar 比率:    {self.calmar_ratio:.4f}",
            f"    月度胜率:       {self.win_rate*100:.1f}%",
            f"    平均换手率:     {self.avg_turnover*100:.1f}%",
            f"    信息比率:       {self.information_ratio:.4f}",
            "",
            "  风险指标:",
            f"    最大回撤期间:   见 drawdown_series",
            f"    正收益月数:      {(self.monthly_returns > 0).sum()} / {self.n_months}",
            f"    负收益月数:      {(self.monthly_returns < 0).sum()} / {self.n_months}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame({
            "date": self.dates,
            "cumulative_return": self.cumulative_returns,
            "monthly_return": list(self.monthly_returns) + [np.nan] * (
                len(self.dates) - len(self.monthly_returns)
            ),
            "benchmark_cumulative": (
                self.benchmark_cumulative
                if len(self.benchmark_cumulative) == len(self.dates)
                else [np.nan] * len(self.dates)
            ),
            "drawdown": (
                self.drawdown_series
                if len(self.drawdown_series) == len(self.dates)
                else [np.nan] * len(self.dates)
            ),
        })

    def plot(self, save_path: Optional[str] = None):
        """绘制回测图表 (需要 matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            # 1. 资金曲线
            ax = axes[0]
            ax.plot(self.dates, self.cumulative_returns, label="RL Strategy", linewidth=1.5)
            if len(self.benchmark_cumulative) == len(self.dates):
                ax.plot(
                    self.dates, self.benchmark_cumulative,
                    label="Benchmark (Equal Weight)", linewidth=1, alpha=0.7, linestyle="--"
                )
            ax.set_ylabel("Cumulative Return")
            ax.set_title("RL Stock Selector — Performance")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # 2. 回撤曲线
            ax = axes[1]
            ax.fill_between(
                range(len(self.drawdown_series)),
                0, self.drawdown_series,
                color="red", alpha=0.3, label="Drawdown"
            )
            ax.set_ylabel("Drawdown")
            ax.set_title("Drawdown Curve")
            ax.grid(True, alpha=0.3)

            # 3. 月度收益分布
            ax = axes[2]
            ax.bar(
                range(len(self.monthly_returns)),
                self.monthly_returns,
                color=["green" if r > 0 else "red" for r in self.monthly_returns],
                alpha=0.7,
            )
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_ylabel("Monthly Return")
            ax.set_xlabel("Month")
            ax.set_title("Monthly Returns Distribution")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("  Warning: matplotlib not available, skipping plot")


def compute_metrics(
    monthly_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """
    计算绩效指标

    Parameters
    ----------
    monthly_returns : np.ndarray (n_months,)
        月度收益率序列
    benchmark_returns : np.ndarray (n_months,) or None
        基准收益率序列
    risk_free_rate : float
        年化无风险利率

    Returns
    -------
    metrics : dict
    """
    if len(monthly_returns) == 0:
        return {}

    n_months = len(monthly_returns)
    monthly_rf = risk_free_rate / 12

    # 累积净值
    cumulative = np.cumprod(1 + monthly_returns)
    total_return = cumulative[-1]

    # 年化收益
    annual_return = total_return ** (12.0 / n_months) - 1

    # 年化波动率
    annual_vol = np.std(monthly_returns, ddof=1) * np.sqrt(12)

    # 夏普比率
    sharpe = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol > 0 else 0.0
    )

    # 最大回撤
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = np.min(drawdown)

    # Calmar 比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    # 胜率
    win_rate = (monthly_returns > 0).mean()

    # 信息比率 (超额收益 / 跟踪误差)
    if benchmark_returns is not None and len(benchmark_returns) == n_months:
        excess = monthly_returns - benchmark_returns
        ir = np.mean(excess) / (np.std(excess, ddof=1) + 1e-8) * np.sqrt(12)
    else:
        ir = 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "information_ratio": float(ir),
        "n_months": n_months,
    }


if HAS_TORCH:

    def run_backtest(
        model,
        env,
        df: pd.DataFrame,
        risk_free_rate: float = 0.03,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        运行回测

        Parameters
        ----------
        model : PPOModel
            训练好的 PPO 模型
        env : StockSelectionEnv
            选股环境 (用于获取日期和股票数据)
        df : pd.DataFrame
            全量数据
        risk_free_rate : float
            年化无风险利率
        deterministic : bool
            是否确定性选股 (True: top_k, False: 采样)
        verbose : bool
            是否打印进度

        Returns
        -------
        BacktestResult
        """
        device = next(model.parameters()).device
        model.eval()

        dates = env.dates[env.lookback:]  # 跳过 lookback 窗口
        monthly_returns = []
        cumulative = [1.0]
        drawdown_series = [0.0]
        positions_history = []
        peak = 1.0

        # 基准: 等权全市场
        benchmark_returns = []
        benchmark_cumulative = [1.0]

        env.reset()

        for i, date in enumerate(dates):
            month_df = df[df["交易日期"] == date]

            if len(month_df) == 0:
                monthly_returns.append(0.0)
                cumulative.append(cumulative[-1])
                benchmark_returns.append(0.0)
                benchmark_cumulative.append(benchmark_cumulative[-1])
                continue

            # 获取当前状态
            state = env._extract_state(month_df)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_indices, _, _ = model.get_action(
                    state_tensor, deterministic=deterministic
                )
            action_np = action_indices.cpu().numpy().flatten()

            # 获取选中的股票代码
            stock_codes = month_df["股票代码"].tolist()
            n_available = len(stock_codes)
            valid_indices = action_np[action_np < n_available]
            selected_codes = [stock_codes[idx] for idx in valid_indices]

            # 计算该月收益
            month_rets = []
            for code in selected_codes:
                stock_row = month_df[month_df["股票代码"] == code]
                if len(stock_row) == 0:
                    continue
                row = stock_row.iloc[0]

                daily_rets = row.get("下周期每天涨跌幅", [])
                if isinstance(daily_rets, str):
                    import ast
                    try:
                        daily_rets = ast.literal_eval(daily_rets)
                    except (ValueError, SyntaxError):
                        daily_rets = []
                if isinstance(daily_rets, list) and len(daily_rets) > 0:
                    cum_ret = np.prod([1 + r for r in daily_rets])
                    month_rets.append(cum_ret - 1)
                else:
                    month_rets.append(0.0)

            portfolio_return = np.mean(month_rets) if month_rets else 0.0
            portfolio_return -= 0.0012  # 交易成本扣除

            monthly_returns.append(portfolio_return)

            # 更新资金曲线
            new_value = cumulative[-1] * (1 + portfolio_return)
            cumulative.append(new_value)
            if new_value > peak:
                peak = new_value
            dd = (new_value - peak) / peak
            drawdown_series.append(dd)

            # 持仓记录
            positions_history.append({
                "date": str(date)[:10],
                "selected_codes": selected_codes,
                "return": portfolio_return,
                "n_selected": len(selected_codes),
            })

            # 基准: 等权全市场收益
            if "涨跌幅" in month_df.columns:
                bench_ret = month_df["涨跌幅"].mean()
            else:
                bench_ret = 0.0
            benchmark_returns.append(bench_ret)
            benchmark_cumulative.append(benchmark_cumulative[-1] * (1 + bench_ret))

            if verbose and (i + 1) % 24 == 0:
                print(
                    f"  Backtest {i+1:4d}/{len(dates)} | "
                    f"Cumulative: {cumulative[-1]:.4f} | "
                    f"Drawdown: {drawdown_series[-1]*100:.1f}% | "
                    f"Return: {portfolio_return*100:+.2f}%"
                )

        # 计算指标
        monthly_returns_arr = np.array(monthly_returns)
        benchmark_arr = np.array(benchmark_returns)
        metrics = compute_metrics(monthly_returns_arr, benchmark_arr, risk_free_rate)

        # 计算换手率
        turnovers = []
        for i in range(1, len(positions_history)):
            prev_codes = set(positions_history[i - 1]["selected_codes"])
            curr_codes = set(positions_history[i]["selected_codes"])
            if len(curr_codes) > 0:
                turnover = 1 - len(prev_codes & curr_codes) / len(curr_codes)
                turnovers.append(turnover)
        avg_turnover = np.mean(turnovers) if turnovers else 0.0

        return BacktestResult(
            cumulative_returns=np.array(cumulative[1:]),  # skip initial 1.0
            monthly_returns=monthly_returns_arr,
            benchmark_cumulative=np.array(benchmark_cumulative[1:]),
            drawdown_series=np.array(drawdown_series[1:]),
            dates=[str(d)[:10] for d in dates],
            positions_history=positions_history,
            total_return=metrics.get("total_return", 0.0),
            annual_return=metrics.get("annual_return", 0.0),
            annual_volatility=metrics.get("annual_volatility", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            calmar_ratio=metrics.get("calmar_ratio", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            avg_turnover=avg_turnover,
            information_ratio=metrics.get("information_ratio", 0.0),
            n_months=metrics.get("n_months", 0),
        )


else:
    def run_backtest(*args, **kwargs):
        raise ImportError("torch is required for backtesting")


def main():
    """示例回测入口"""
    print("""
    RL Stock Selector — Backtest Module
    ====================================

    Usage:
        from rl_stock_selector.backtest import run_backtest, BacktestResult

        # 加载模型
        model = PPOModel(state_dim=256, action_dim=6, n_stocks=100)
        model.load_state_dict(torch.load("ppo_model.pt")["model_state_dict"])

        # 运行回测
        result = run_backtest(model, env, df, deterministic=True)
        print(result.summary())
        result.plot("backtest_results.png")
    """)


if __name__ == "__main__":
    main()
