"""
Dynamic position sizing + best sell-side strategies
Tests: equal weight, rank weight, signal strength, inverse vol, Kelly-style
Combines with best take profit from sell-side test
"""
import ast
import pandas as pd
import numpy as np

pd.set_option('expand_frame_repr', False)

select_stock_num = 6
c_rate = 1.2 / 10000
t_rate = 1 / 1000
sell_cost = c_rate + t_rate

# === Load data ===
df = pd.read_csv('stock_data.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)

# === Apply same filters as baseline ===
df = df[df['上市至今交易天数'] > 250]
df = df[~df['股票代码'].str.contains('bj')]

ind_col = '新版申万二级行业名称'
ind_val = df.groupby([ind_col, '交易日期']).agg(
    med_ep=('市盈率倒数', 'median'),
    med_bp=('市净率倒数', 'median'),
).reset_index()

def calc_val_percentile(grp):
    grp = grp.sort_values('交易日期')
    ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
    bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
    grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
    return grp

ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
df = df.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
df['val_pct'] = df['val_pct'].fillna(0.5)
df = df[df['val_pct'] < 0.68]

cutoff = df.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(0.52))
df = df[df['bias_20'] < cutoff]

df['因子'] = df['总市值']
df['排名'] = df.groupby('交易日期')['因子'].rank()
df = df[df['排名'] <= select_stock_num]

df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))

print(f"Selected {len(df)} stock-period observations")


# === Sell-side functions ===
def apply_no_side(daily_returns):
    return list(daily_returns), False

def apply_take_profit(daily_returns, tp_pct):
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 > tp_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered

def apply_stop_loss(daily_returns, sl_pct):
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 < -sl_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


# === Position sizing schemes ===

def weights_equal(n):
    """Equal weight"""
    return np.ones(n) / n

def weights_rank(n):
    """Linear rank weight: best rank (smallest cap) gets highest weight"""
    w = np.arange(n, 0, -1, dtype=float)
    return w / w.sum()

def weights_rank_squared(n):
    """Squared rank weight: more aggressive"""
    w = np.arange(n, 0, -1, dtype=float) ** 2
    return w / w.sum()

def weights_inverse_market_cap(market_caps):
    """Inverse market cap weight within selected stocks"""
    w = 1.0 / np.array(market_caps, dtype=float)
    return w / w.sum()

def weights_bias_signal(bias_values):
    """Weight by signal strength: lower bias (more oversold) = higher weight"""
    # bias_20 can be negative. Lower = further below MA = more oversold.
    # Transform: -bias (higher for more negative bias)
    arr = np.array(bias_values, dtype=float)
    w = np.clip(-arr + 0.1, 0.01, None)  # shift to ensure positivity
    return w / w.sum()

def weights_composite(n, market_caps, bias_values):
    """Composite: rank * inverse cap * bias signal"""
    r_w = np.arange(n, 0, -1, dtype=float)
    cap_w = 1.0 / np.array(market_caps, dtype=float)
    cap_w = cap_w / cap_w.mean()  # normalize
    bias_w = np.clip(-np.array(bias_values, dtype=float) + 0.1, 0.01, None)
    bias_w = bias_w / bias_w.mean()
    w = r_w * cap_w * bias_w
    return w / w.sum()


# === Evaluate ===
def evaluate(curves):
    df_eval = pd.DataFrame(curves, columns=['交易日期', '选股下周期涨跌幅'])
    df_eval['累积净值'] = (df_eval['选股下周期涨跌幅'] + 1).cumprod()
    final_nav = df_eval['累积净值'].iloc[-1]
    days = (df_eval['交易日期'].iloc[-1] - df_eval['交易日期'].iloc[0]).days
    annual_return = final_nav ** (365.0 / days) - 1
    df_eval['max2here'] = df_eval['累积净值'].expanding().max()
    df_eval['dd2here'] = df_eval['累积净值'] / df_eval['max2here'] - 1
    max_draw_down = df_eval['dd2here'].min()
    calmar = annual_return / abs(max_draw_down)
    return {
        '年化': f'{annual_return*100:.1f}%',
        '净值': f'{final_nav:.0f}',
        'MDD': f'{max_draw_down*100:.1f}%',
        'Calmar': f'{calmar:.2f}',
        '年化_raw': annual_return,
        'MDD_raw': max_draw_down,
        'Calmar_raw': calmar,
    }


all_results = []


def run_position_test(name, side_func, side_args, weight_func, weight_needs_data=False):
    """Run a strategy with given side function and position weighting"""
    curves = []
    for date, grp in df.groupby('交易日期'):
        n_stocks = len(grp)
        if n_stocks == 0:
            continue

        daily_lists = list(grp['下周期每天涨跌幅'])

        # Compute per-stock period returns
        stock_rets = []
        for daily_ret in daily_lists:
            modified, triggered = side_func(daily_ret, *side_args)
            cumret = np.prod([1 + r for r in modified])
            if not triggered:
                cumret *= (1 - sell_cost)
            stock_rets.append(cumret)

        # Compute weights
        if weight_needs_data:
            if weight_func == weights_inverse_market_cap:
                w = weight_func(list(grp['总市值']))
            elif weight_func == weights_bias_signal:
                w = weight_func(list(grp['bias_20']))
            elif weight_func == weights_composite:
                w = weight_func(n_stocks, list(grp['总市值']), list(grp['bias_20']))
            else:
                w = weight_func(n_stocks)
        else:
            w = weight_func(n_stocks)

        # Weighted portfolio return
        portfolio_ret = np.dot(w, np.array(stock_rets) - 1) + 1  # weighted average of final values
        portfolio_ret *= (1 - c_rate)  # buy cost
        period_ret = portfolio_ret - 1
        curves.append((date, period_ret))

    return curves


# === Get baseline for each side strategy with equal weight ===
print("Computing baselines...")

# Baseline: no side, equal weight
curves = run_position_test('baseline', apply_no_side, (), weights_equal)
baseline_r = evaluate(curves)
print(f"\n{'='*70}")
print(f"BASELINE (equal weight, no side):")
print(f"  年化{baseline_r['年化']}  净值{baseline_r['净值']}  MDD{baseline_r['MDD']}  Calmar{baseline_r['Calmar']}")

# Also baseline with best take profit, equal weight
curves_tp30 = run_position_test('tp30_eq', apply_take_profit, (0.30,), weights_equal)
tp30_bl = evaluate(curves_tp30)
print(f"\nTAKE PROFIT 30% equal weight:")
print(f"  年化{tp30_bl['年化']}  净值{tp30_bl['净值']}  MDD{tp30_bl['MDD']}  Calmar{tp30_bl['Calmar']}")

curves_tp25 = run_position_test('tp25_eq', apply_take_profit, (0.25,), weights_equal)
tp25_bl = evaluate(curves_tp25)
print(f"\nTAKE PROFIT 25% equal weight:")
print(f"  年化{tp25_bl['年化']}  净值{tp25_bl['净值']}  MDD{tp25_bl['MDD']}  Calmar{tp25_bl['Calmar']}")

print(f"\n{'='*70}")
print("TESTING DYNAMIC POSITION SIZING")
print(f"{'='*70}")


# === Test all combinations ===
side_configs = [
    ("无卖出优化", apply_no_side, ()),
    ("止盈20%", apply_take_profit, (0.20,)),
    ("止盈25%", apply_take_profit, (0.25,)),
    ("止盈30%", apply_take_profit, (0.30,)),
    ("止损10%+止盈30%", apply_stop_loss, (0.10,)),  # We'll test combined separately
]

weight_configs = [
    ("等权", weights_equal, False),
    ("排名加权", weights_rank, False),
    ("排名²加权", weights_rank_squared, False),
    ("1/市值加权", weights_inverse_market_cap, True),
    ("信号加权(bias)", weights_bias_signal, True),
    ("复合加权", weights_composite, True),
]

for side_name, side_func, side_args in side_configs:
    print(f"\n--- {side_name} ---")
    for weight_name, weight_func, needs_data in weight_configs:
        curves = run_position_test(
            f'{side_name}+{weight_name}',
            side_func, side_args,
            weight_func, needs_data
        )
        r = evaluate(curves)
        r['策略'] = f'{side_name}+{weight_name}'
        all_results.append(r)
        imp = r['年化_raw'] - baseline_r['年化_raw']
        print(f"  {r['策略']:28s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']:>6s}  Δ{imp*100:+.1f}pp")

# === Special: best sell-side combos with rank weighting ===
print(f"\n--- Best Combos with Rank Weighting ---")

# Take profit 30% + rank² weight
curves = run_position_test('tp30+rank2', apply_take_profit, (0.30,), weights_rank_squared)
r = evaluate(curves)
r['策略'] = '止盈30%+排名²加权'
all_results.append(r)
print(f"  {r['策略']:28s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']:>6s}")

# Take profit 25% + rank² weight
curves = run_position_test('tp25+rank2', apply_take_profit, (0.25,), weights_rank_squared)
r = evaluate(curves)
r['策略'] = '止盈25%+排名²加权'
all_results.append(r)
print(f"  {r['策略']:28s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']:>6s}")

# Take profit 20% + rank² weight
curves = run_position_test('tp20+rank2', apply_take_profit, (0.20,), weights_rank_squared)
r = evaluate(curves)
r['策略'] = '止盈20%+排名²加权'
all_results.append(r)
print(f"  {r['策略']:28s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']:>6s}")

# Stop loss + Take profit with rank weighting
def apply_stop_loss_take_profit(daily_returns, sl_pct, tp_pct):
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 < -sl_pct or cumret - 1 > tp_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered

# Best combos from sell-side: 止损10%+止盈30% and 止损10%+止盈25%
for sl, tp, wt_name, wt_func in [
    (0.10, 0.30, '等权', weights_equal),
    (0.10, 0.30, '排名加权', weights_rank),
    (0.10, 0.30, '排名²加权', weights_rank_squared),
    (0.10, 0.25, '等权', weights_equal),
    (0.10, 0.25, '排名加权', weights_rank),
    (0.10, 0.25, '排名²加权', weights_rank_squared),
]:
    curves = run_position_test(
        f'sl{tp*100:.0f}_tp{tp*100:.0f}_{wt_name}',
        apply_stop_loss_take_profit, (sl, tp),
        wt_func, wt_func in (weights_inverse_market_cap, weights_bias_signal, weights_composite)
    )
    r = evaluate(curves)
    r['策略'] = f'止损{sl*100:.0f}%+止盈{tp*100:.0f}%+{wt_name}'
    all_results.append(r)
    imp = r['年化_raw'] - baseline_r['年化_raw']
    print(f"  {r['策略']:28s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']:>6s}  Δ{imp*100:+.1f}pp")


# === FINAL RANKINGS ===
print(f"\n\n{'='*70}")
print("FINAL RANKINGS — TOP 20 BY ANNUALIZED RETURN")
print(f"Baseline: 年化{baseline_r['年化']}  Calmar{baseline_r['Calmar']}")
print(f"{'='*70}")
results_df = pd.DataFrame(all_results)
for i, row in results_df.sort_values('年化_raw', ascending=False).head(20).iterrows():
    flag = " ***" if row['年化_raw'] > baseline_r['年化_raw'] else ""
    print(f"  {row['策略']:30s} 年化{row['年化']:>10s}  MDD{row['MDD']:>10s}  Calmar{row['Calmar']:>6s}{flag}")

print(f"\n{'='*70}")
print("FINAL RANKINGS — TOP 20 BY CALMAR RATIO")
print(f"{'='*70}")
for i, row in results_df.sort_values('Calmar_raw', ascending=False).head(20).iterrows():
    flag = " ***" if row['Calmar_raw'] > baseline_r['Calmar_raw'] else ""
    print(f"  {row['策略']:30s} 年化{row['年化']:>10s}  MDD{row['MDD']:>10s}  Calmar{row['Calmar']:>6s}{flag}")

results_df.to_csv('position_sizing_results.csv', encoding='gbk', index=False)
print(f"\nResults saved to position_sizing_results.csv")
