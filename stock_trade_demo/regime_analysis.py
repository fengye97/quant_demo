"""
Market regime analysis: quarterly bull/bear detection + adaptive strategy testing
"""
import pandas as pd, numpy as np, ast

pd.set_option('expand_frame_repr', False)

# === Load data ===
df = pd.read_csv('stock_data.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))

# === Build market proxy ===
mkt = df.groupby('交易日期').agg(
    mkt_ret=('涨跌幅', 'mean'),
    up_pct=('涨跌幅', lambda x: (x > 0).mean()),  # % of stocks up
).sort_index()

# === Regime indicators ===
# 1. 12-month moving average of cumulative market return
mkt['mkt_cum'] = (1 + mkt['mkt_ret']).cumprod()
mkt['ma3'] = mkt['mkt_cum'].rolling(3).mean()
mkt['ma6'] = mkt['mkt_cum'].rolling(6).mean()
mkt['ma12'] = mkt['mkt_cum'].rolling(12).mean()

# 2. Quarterly (3-month) market return
mkt['qtr_ret'] = mkt['mkt_cum'] / mkt['mkt_cum'].shift(3) - 1

# 3. Drawdown from 12-month peak
mkt['peak12'] = mkt['mkt_cum'].rolling(12).max()
mkt['dd_12'] = mkt['mkt_cum'] / mkt['peak12'] - 1

# 4. Regime definitions
mkt['regime_ma'] = np.where(mkt['mkt_cum'] > mkt['ma12'], 'bull', 'bear')
mkt['regime_qtr'] = np.where(mkt['qtr_ret'] > 0, 'bull', 'bear')
mkt['regime_dd'] = np.where(mkt['dd_12'] > -0.10, 'bull', 'bear')  # within 10% of peak
mkt['regime_combo'] = np.where(
    (mkt['mkt_cum'] > mkt['ma12']) & (mkt['qtr_ret'] > -0.05), 'bull',
    np.where((mkt['mkt_cum'] < mkt['ma12']) & (mkt['qtr_ret'] < 0.05), 'bear', 'neutral')
)

print("Market regime distribution:")
for col in ['regime_ma', 'regime_qtr', 'regime_dd', 'regime_combo']:
    print(f"  {col}: {mkt[col].value_counts().to_dict()}")

# === Merge regime into stock data ===
df = df.merge(mkt[['regime_ma', 'regime_qtr', 'regime_dd', 'regime_combo', 'qtr_ret', 'dd_12']],
              left_on='交易日期', right_index=True, how='left')

# === Apply filters (same as baseline) ===
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
df = df[df['排名'] <= 6]

c_rate = 1.2 / 10000
sell_cost = c_rate + 1/1000

print(f"\nSelected {len(df)} stock-period observations")


# === Strategy functions ===

def apply_tp(daily_returns, tp_pct):
    cumret = 1.0; result = []; triggered = False
    for r in daily_returns:
        if triggered: result.append(0.0); continue
        cumret *= (1 + r); result.append(r)
        if cumret - 1 > tp_pct: triggered = True; result[-1] = result[-1] - sell_cost
    return result, triggered


def compute_period_returns(grp, tp_pct, n_stocks=None):
    """Compute period return with take profit and optional stock count override"""
    daily_lists = list(grp['下周期每天涨跌幅'])
    if n_stocks and n_stocks < len(daily_lists):
        daily_lists = daily_lists[:n_stocks]  # take top N by rank (smallest)

    final_rets = []
    for dl in daily_lists:
        modified, triggered = apply_tp(dl, tp_pct)
        cr = np.prod([1 + r for r in modified])
        if not triggered: cr *= (1 - sell_cost)
        final_rets.append(cr)
    return np.mean(final_rets) * (1 - c_rate) - 1


def evaluate_curves(curves, label=''):
    df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
    df_e['nav'] = (df_e['ret'] + 1).cumprod()
    days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
    ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
    df_e['max2here'] = df_e['nav'].expanding().max()
    df_e['dd'] = df_e['nav'] / df_e['max2here'] - 1
    mdd = df_e['dd'].min()
    calmar = ann / abs(mdd)
    print(f"  {label}: 年化{ann*100:.1f}% 净值{df_e.nav.iloc[-1]:.0f} MDD{mdd*100:.1f}% Calmar{calmar:.2f}")
    return ann, mdd, calmar, df_e['nav'].iloc[-1]


# ================================================================
# PART 1: Analyze strategy performance in each regime
# ================================================================
print("\n" + "=" * 60)
print("PART 1: STRATEGY PERFORMANCE BY MARKET REGIME")
print("=" * 60)

for regime_col in ['regime_ma', 'regime_qtr', 'regime_combo']:
    print(f"\n--- {regime_col} ---")
    for regime_name in ['bull', 'bear', 'neutral']:
        regime_dates = set(mkt[mkt[regime_col] == regime_name].index)
        regime_df = df[df['交易日期'].isin(regime_dates)]

        if len(regime_df) == 0:
            continue

        curves = []
        for date, grp in regime_df.groupby('交易日期'):
            ret = compute_period_returns(grp, 0.32)
            curves.append((date, ret))

        n_periods = len(curves)
        if n_periods > 0:
            ann, mdd, cal, nav = evaluate_curves(curves, f'{regime_name} ({n_periods}期)')

# ================================================================
# PART 2: Regime-specific parameter optimization
# ================================================================
print("\n" + "=" * 60)
print("PART 2: REGIME-ADAPTIVE STRATEGY")
print("=" * 60)

regime_col = 'regime_ma'  # MA-based regime as primary

# Get regime dates
bull_dates = set(mkt[mkt[regime_col] == 'bull'].index)
bear_dates = set(mkt[mkt[regime_col] == 'bear'].index)

# Baseline: uniform TP32%
curves_bl = []
for date, grp in df.groupby('交易日期'):
    ret = compute_period_returns(grp, 0.32)
    curves_bl.append((date, ret))
ann_bl, mdd_bl, cal_bl, nav_bl = evaluate_curves(curves_bl, 'Uniform TP32%')
print()

# Grid search: different TP and stock count per regime
# Hypothesis: in bull markets, higher TP and more stocks; in bear, lower TP and fewer stocks
results = []

for bull_tp in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]:
    for bear_tp in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
        for bull_n in [6]:  # keep 6 stocks in bull
            for bear_n in [3, 4, 5, 6]:  # fewer stocks in bear
                curves = []
                for date, grp in df.groupby('交易日期'):
                    if date in bull_dates:
                        ret = compute_period_returns(grp, bull_tp, bull_n)
                    elif date in bear_dates:
                        ret = compute_period_returns(grp, bear_tp, bear_n)
                    else:
                        ret = compute_period_returns(grp, 0.32)
                    curves.append((date, ret))

                df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
                df_e['nav'] = (df_e['ret'] + 1).cumprod()
                days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
                ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
                df_e['max2here'] = df_e['nav'].expanding().max()
                df_e['dd'] = df_e['nav'] / df_e['max2here'] - 1
                mdd = df_e['dd'].min()
                calmar = ann / abs(mdd)

                tp_str = f'Bull_TP{bull_tp*100:.0f}%/Bear_TP{bear_tp*100:.0f}%/N{bear_n}'
                results.append({
                    '策略': tp_str,
                    '年化': ann, 'MDD': mdd, 'Calmar': calmar,
                    '净值': df_e['nav'].iloc[-1],
                    'bull_tp': bull_tp, 'bear_tp': bear_tp, 'bear_n': bear_n,
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('年化', ascending=False)

print(f"\nTop 20 regime-adaptive strategies (vs uniform TP32%: {ann_bl*100:.1f}%):")
for i, row in results_df.head(20).iterrows():
    imp = row['年化'] - ann_bl
    flag = " ***" if row['年化'] > ann_bl else ""
    print(f"  {row['策略']:42s} 年化{row['年化']*100:5.1f}%  MDD{row['MDD']*100:5.1f}%  Calmar{row['Calmar']:.2f}  Δ{imp*100:+.1f}pp{flag}")

print(f"\nBest Calmar strategies:")
for i, row in results_df.sort_values('Calmar', ascending=False).head(10).iterrows():
    print(f"  {row['策略']:42s} 年化{row['年化']*100:5.1f}%  MDD{row['MDD']*100:5.1f}%  Calmar{row['Calmar']:.2f}")

# ================================================================
# PART 3: Regime analysis - when does TP help most?
# ================================================================
print("\n" + "=" * 60)
print("PART 3: TAKE PROFIT EFFECTIVENESS BY REGIME")
print("=" * 60)

for regime_col in ['regime_ma', 'regime_qtr']:
    print(f"\n--- {regime_col} ---")
    for regime_name in ['bull', 'bear']:
        regime_dates = set(mkt[mkt[regime_col] == regime_name].index)
        regime_df = df[df['交易日期'].isin(regime_dates)]

        if len(regime_df) == 0:
            continue

        # No TP
        curves_no = []
        curves_tp32 = []
        for date, grp in regime_df.groupby('交易日期'):
            ret_no = compute_period_returns(grp, 99.0)  # never triggers
            ret_tp32 = compute_period_returns(grp, 0.32)
            curves_no.append((date, ret_no))
            curves_tp32.append((date, ret_tp32))

        ann_no, _, _, _ = evaluate_curves(curves_no, f'{regime_name} 无TP')
        ann_tp, _, _, _ = evaluate_curves(curves_tp32, f'{regime_name} TP32%')
        print(f"    止盈提升: {ann_tp*100 - ann_no*100:+.1f}pp 年化")
