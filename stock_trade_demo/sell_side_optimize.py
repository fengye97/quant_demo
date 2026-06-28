"""
Sell-side optimization + dynamic position sizing
Correctly handles transaction costs for triggered vs non-triggered stocks
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

print(f"Selected {len(df)} stock-period observations across {df['交易日期'].nunique()} periods")


# === Side functions - return (modified_daily_returns, triggered) ===

def apply_stop_loss(daily_returns, stop_loss_pct):
    """Returns (modified_returns, triggered)"""
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 < -stop_loss_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost  # sell cost on trigger day
    return result, triggered


def apply_take_profit(daily_returns, take_profit_pct):
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 > take_profit_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


def apply_trailing_stop(daily_returns, trail_pct):
    cumret = 1.0
    peak = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        peak = max(peak, cumret)
        result.append(r)
        if (peak - cumret) / peak > trail_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


def apply_combined(daily_returns, stop_loss_pct, take_profit_pct):
    cumret = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 < -stop_loss_pct or cumret - 1 > take_profit_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


def apply_trailing_and_stop(daily_returns, stop_loss_pct, trail_pct):
    cumret = 1.0
    peak = 1.0
    result = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)
            continue
        cumret *= (1 + r)
        peak = max(peak, cumret)
        result.append(r)
        if cumret - 1 < -stop_loss_pct or (peak - cumret) / peak > trail_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


def apply_baseline(daily_returns):
    """No sell optimization - hold to end"""
    return list(daily_returns), False


# === Compute period returns correctly ===

def compute_period_return(daily_lists, side_func, *args):
    """
    For a list of stock daily returns (one period), apply side function
    and return the average portfolio return for the period.
    Sell cost: applied in side_func for triggered stocks, at end for non-triggered.
    """
    final_returns = []
    for daily_returns in daily_lists:
        modified, triggered = side_func(daily_returns, *args) if args else side_func(daily_returns)
        cumret = np.prod([1 + r for r in modified])
        if not triggered:
            cumret *= (1 - sell_cost)  # sell cost at period end for non-triggered
        final_returns.append(cumret)

    avg_return = np.mean(final_returns)
    avg_return *= (1 - c_rate)  # buy cost
    return avg_return - 1


# === Prepare period data ===
period_data = []
for date, grp in df.groupby('交易日期'):
    daily_returns_lists = list(grp['下周期每天涨跌幅'])
    period_data.append((date, daily_returns_lists))

print(f"Periods: {len(period_data)}")


# === Evaluation ===
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


# === BASELINE ===
baseline_curves = []
for date, rets in period_data:
    per_ret = compute_period_return(rets, apply_baseline)
    baseline_curves.append((date, per_ret))
baseline_r = evaluate(baseline_curves)
print(f"\n{'='*65}")
print(f"BASELINE: 年化{baseline_r['年化']}  净值{baseline_r['净值']}  MDD{baseline_r['MDD']}  Calmar{baseline_r['Calmar']}")
print(f"{'='*65}")

all_results = []


# === 1. STOP LOSS ===
print(f"\n--- Stop Loss (止损) ---")
for sl in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    curves = []
    for date, rets in period_data:
        curves.append((date, compute_period_return(rets, apply_stop_loss, sl)))
    r = evaluate(curves)
    r['策略'] = f'止损{sl*100:.0f}%'
    all_results.append(r)
    print(f"  {r['策略']:12s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']}")


# === 2. TAKE PROFIT ===
print(f"\n--- Take Profit (止盈) ---")
for tp in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    curves = []
    for date, rets in period_data:
        curves.append((date, compute_period_return(rets, apply_take_profit, tp)))
    r = evaluate(curves)
    r['策略'] = f'止盈{tp*100:.0f}%'
    all_results.append(r)
    print(f"  {r['策略']:12s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']}")


# === 3. TRAILING STOP ===
print(f"\n--- Trailing Stop (移动止损) ---")
for trail in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    curves = []
    for date, rets in period_data:
        curves.append((date, compute_period_return(rets, apply_trailing_stop, trail)))
    r = evaluate(curves)
    r['策略'] = f'移动止损{trail*100:.0f}%'
    all_results.append(r)
    print(f"  {r['策略']:12s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']}")


# === 4. STOP LOSS + TAKE PROFIT ===
print(f"\n--- Combined Stop Loss + Take Profit ---")
combos = [
    (0.05, 0.15), (0.05, 0.20), (0.05, 0.25),
    (0.08, 0.15), (0.08, 0.20), (0.08, 0.25), (0.08, 0.30),
    (0.10, 0.20), (0.10, 0.25), (0.10, 0.30),
    (0.12, 0.20), (0.12, 0.25), (0.12, 0.30), (0.12, 0.35),
]
for sl, tp in combos:
    curves = []
    for date, rets in period_data:
        curves.append((date, compute_period_return(rets, apply_combined, sl, tp)))
    r = evaluate(curves)
    r['策略'] = f'止损{sl*100:.0f}%+止盈{tp*100:.0f}%'
    all_results.append(r)
    print(f"  {r['策略']:18s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']}")


# === 5. TRAILING + STOP ===
print(f"\n--- Trailing Stop + Stop Loss ---")
tcombos = [
    (0.05, 0.08), (0.05, 0.10), (0.05, 0.12),
    (0.08, 0.10), (0.08, 0.12), (0.08, 0.15),
    (0.10, 0.12), (0.10, 0.15), (0.10, 0.20),
]
for sl, trail in tcombos:
    curves = []
    for date, rets in period_data:
        curves.append((date, compute_period_return(rets, apply_trailing_and_stop, sl, trail)))
    r = evaluate(curves)
    r['策略'] = f'止损{sl*100:.0f}%+移动{trail*100:.0f}%'
    all_results.append(r)
    print(f"  {r['策略']:18s} 年化{r['年化']:>10s}  净值{r['净值']:>10s}  MDD{r['MDD']:>10s}  Calmar{r['Calmar']}")


# === RANKINGS ===
print(f"\n{'='*65}")
print("TOP 15 BY ANNUALIZED RETURN (vs baseline {})".format(baseline_r['年化']))
print(f"{'='*65}")
results_df = pd.DataFrame(all_results)
for i, row in results_df.sort_values('年化_raw', ascending=False).head(15).iterrows():
    flag = " ***" if row['年化_raw'] > baseline_r['年化_raw'] else ""
    print(f"  {row['策略']:20s} 年化{row['年化']:>10s}  MDD{row['MDD']:>10s}  Calmar{row['Calmar']:>6s}{flag}")

print(f"\n{'='*65}")
print("TOP 15 BY CALMAR RATIO (vs baseline {})".format(baseline_r['Calmar']))
print(f"{'='*65}")
for i, row in results_df.sort_values('Calmar_raw', ascending=False).head(15).iterrows():
    flag = " ***" if row['Calmar_raw'] > baseline_r['Calmar_raw'] else ""
    print(f"  {row['策略']:20s} 年化{row['年化']:>10s}  MDD{row['MDD']:>10s}  Calmar{row['Calmar']:>6s}{flag}")

results_df.to_csv('sell_side_results.csv', encoding='gbk', index=False)
print(f"\nResults saved to sell_side_results.csv")
