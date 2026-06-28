"""
Final optimization analysis: find remaining edge
"""
import pandas as pd, numpy as np, ast

pd.set_option('expand_frame_repr', False)

# === Load and prepare ===
df = pd.read_csv('stock_data.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))

# Build market regime
mkt_ret = df.groupby('交易日期')['涨跌幅'].mean()
mkt_cum = (1 + mkt_ret).cumprod()
mkt_ma12 = mkt_cum.rolling(12).mean()
regime_map = (mkt_cum > mkt_ma12).map({True: 'bull', False: 'bear'})

# Market stats for skip rules
mkt_vol = mkt_ret.rolling(6).std()
mkt_dd = mkt_cum / mkt_cum.rolling(12).max() - 1  # 12-month drawdown

df['市场状态'] = df['交易日期'].map(regime_map)

# Filters
df = df[df['上市至今交易天数'] > 250]
df = df[~df['股票代码'].str.contains('bj')]

c_rate = 1.2 / 10000
sell_cost = c_rate + 1/1000

ind_col = '新版申万二级行业名称'

def apply_filters(data, val_cut=0.68, bias_cut=0.52, regime=None):
    """Apply filters with optional regime-specific params"""
    d = data.copy()

    ind_val = d.groupby([ind_col, '交易日期']).agg(
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
    d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    d['val_pct'] = d['val_pct'].fillna(0.5)

    # Regime-specific val_pct cutoff
    if regime == 'bull':
        vc = val_cut if isinstance(val_cut, (int, float)) else val_cut[0]
    elif regime == 'bear':
        vc = val_cut if isinstance(val_cut, (int, float)) else val_cut[1]
    else:
        vc = val_cut if isinstance(val_cut, (int, float)) else val_cut

    d = d[d['val_pct'] < vc]

    # Regime-specific bias cutoff
    if regime == 'bull':
        bc = bias_cut if isinstance(bias_cut, (int, float)) else bias_cut[0]
    elif regime == 'bear':
        bc = bias_cut if isinstance(bias_cut, (int, float)) else bias_cut[1]
    else:
        bc = bias_cut if isinstance(bias_cut, (int, float)) else bias_cut

    cutoff = d.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(bc))
    d = d[d['bias_20'] < cutoff]

    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    return d


def apply_tp(daily_returns, tp_pct):
    if tp_pct >= 0.99:
        return list(daily_returns), False
    cumret = 1.0; result = []; triggered = False
    for r in daily_returns:
        if triggered: result.append(0.0); continue
        cumret *= (1 + r); result.append(r)
        if cumret - 1 > tp_pct: triggered = True; result[-1] = result[-1] - sell_cost
    return result, triggered


def run_strategy(data, bull_tp=0.30, bear_tp=0.22, bull_n=6, bear_n=4,
                 val_cut=0.68, bias_cut=0.52, skip_vol=None, skip_dd=None):
    """Run strategy with given parameters, return period curves"""
    curves = []
    skipped = 0
    for date, grp in data.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]

        # Skip rules
        if skip_vol is not None:
            vol = mkt_vol.get(date, 0)
            if vol > skip_vol:
                curves.append((date, 0.0))
                skipped += 1
                continue

        if skip_dd is not None:
            dd = mkt_dd.get(date, 0)
            if dd < -skip_dd:
                curves.append((date, 0.0))
                skipped += 1
                continue

        if regime == 'bull':
            tp, n_stocks = bull_tp, bull_n
        else:
            tp, n_stocks = bear_tp, bear_n

        daily_lists = list(grp['下周期每天涨跌幅'])
        daily_lists = daily_lists[:n_stocks]

        final_rets = []
        for dl in daily_lists:
            modified, triggered = apply_tp(dl, tp)
            cr = np.prod([1 + r for r in modified])
            if not triggered: cr *= (1 - sell_cost)
            final_rets.append(cr)
        ret = np.mean(final_rets) * (1 - c_rate) - 1
        curves.append((date, ret))

    df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
    df_e['nav'] = (df_e['ret'] + 1).cumprod()
    days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
    ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
    df_e['max2here'] = df_e['nav'].expanding().max()
    df_e['dd'] = df_e['nav'] / df_e['max2here'] - 1
    mdd = df_e['dd'].min()
    calmar = ann / abs(mdd)
    return ann, mdd, calmar, df_e['nav'].iloc[-1], df_e, skipped


# ================================================================
# PART 1: Analyze current strategy weaknesses
# ================================================================
print("=" * 60)
print("PART 1: STRATEGY WEAKNESS ANALYSIS")
print("=" * 60)

df_filtered = apply_filters(df.copy())
ann, mdd, cal, nav, curves_df, _ = run_strategy(df_filtered)
print(f"Current: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f}")

# Find worst periods
curves_df['year'] = curves_df['交易日期'].dt.year
yearly = curves_df.groupby('year').agg(
    ret=('ret', lambda x: np.prod(1 + x) - 1),
    n=('ret', 'count'),
    win_rate=('ret', lambda x: (x > 0).mean()),
    avg_ret=('ret', 'mean'),
).sort_values('ret')

print("\nWorst 10 years:")
for yr, row in yearly.head(10).iterrows():
    print(f"  {yr}: 年收益{row['ret']*100:+.1f}%  月数{row['n']:.0f}  胜率{row['win_rate']*100:.0f}%  月均{row['avg_ret']*100:+.1f}%")

print("\nBest 10 years:")
for yr, row in yearly.tail(10).iterrows():
    print(f"  {yr}: 年收益{row['ret']*100:+.1f}%  月数{row['n']:.0f}  胜率{row['win_rate']*100:.0f}%  月均{row['avg_ret']*100:+.1f}%")

# Find worst drawdown periods
curves_df['dd'] = curves_df['nav'] / curves_df['nav'].expanding().max() - 1
worst_dd_idx = curves_df['dd'].idxmin()
print(f"\nWorst drawdown: {curves_df.loc[worst_dd_idx, '交易日期']} at {curves_df.loc[worst_dd_idx, 'dd']*100:.1f}%")
print(f"Market regime at worst DD: {regime_map.get(curves_df.loc[worst_dd_idx, '交易日期'], 'unknown')}")

# DD period analysis
dd_start = curves_df.loc[:worst_dd_idx, 'nav'].idxmax()
print(f"DD start: {curves_df.loc[dd_start, '交易日期']} (nav={curves_df.loc[dd_start, 'nav']:.1f})")
dd_period = curves_df.loc[dd_start:worst_dd_idx]
print(f"DD period months: {len(dd_period)}, regime distribution:")
print(dd_period['交易日期'].map(regime_map).value_counts().to_dict())

# Regime-specific monthly stats
print("\nMonthly stats by regime:")
for regime in ['bull', 'bear']:
    regime_dates = set(mkt_ret[regime_map == regime].index)
    regime_curves = curves_df[curves_df['交易日期'].isin(regime_dates)]
    print(f"  {regime}: 月均{regime_curves['ret'].mean()*100:+.1f}%  胜率{(regime_curves['ret']>0).mean()*100:.0f}%  "
          f"最差月{regime_curves['ret'].min()*100:.1f}%  最好月{regime_curves['ret'].max()*100:.1f}%")


# ================================================================
# PART 2: Test regime-specific filter thresholds
# ================================================================
print("\n" + "=" * 60)
print("PART 2: REGIME-SPECIFIC FILTER THRESHOLDS")
print("=" * 60)

# Test: tighter bias_20 filter in bear markets (more selective about reversal)
# Test: different val_pct thresholds per regime
results = []
baseline_ann = ann

# Vary bear bias cutoff (baseline is 0.52)
for bear_bias in [0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.60]:
    df_test = apply_filters(df.copy())
    ann_t, mdd_t, cal_t, nav_t, _, _ = run_strategy(
        df_test, bear_n=4, bear_tp=0.22, bull_tp=0.30,
        bias_cut=(0.52, bear_bias)  # tuple: (bull_bias, bear_bias)
    )
    # Note: apply_filters doesn't support regime-specific yet, need to re-filter
    # Actually let me do this properly

# Let me do this properly - re-filter with regime-specific params
print("\nRegime-specific bias_20 cutoff (bear only):")
for bear_bias in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    df_test = df.copy()
    # Apply standard filters
    df_test = df_test[df_test['上市至今交易天数'] > 250]
    df_test = df_test[~df_test['股票代码'].str.contains('bj')]

    ind_val = df_test.groupby([ind_col, '交易日期']).agg(
        med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

    def calc_val_percentile(grp):
        grp = grp.sort_values('交易日期')
        ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
        bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
        grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
        return grp

    ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
    df_test = df_test.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    df_test['val_pct'] = df_test['val_pct'].fillna(0.5)

    # Regime-specific val_pct
    bull_mask = (df_test['市场状态'] == 'bull') & (df_test['val_pct'] >= 0.68)
    bear_mask = (df_test['市场状态'] == 'bear') & (df_test['val_pct'] >= 0.68)
    df_test = df_test[~(bull_mask | bear_mask)]

    # Regime-specific bias_20
    for regime, bias_cut in [('bull', 0.52), ('bear', bear_bias)]:
        regime_data = df_test[df_test['市场状态'] == regime]
        if len(regime_data) > 0:
            cutoff = regime_data.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(bias_cut))
            df_test = df_test[
                ~((df_test['市场状态'] == regime) & (df_test['bias_20'] >= cutoff))
            ]

    df_test['因子'] = df_test['总市值']
    df_test['排名'] = df_test.groupby('交易日期')['因子'].rank()
    df_test = df_test[df_test['排名'] <= 6]

    if len(df_test) > 0:
        ann_t, mdd_t, cal_t, nav_t, _, _ = run_strategy(df_test)
        imp = ann_t - baseline_ann
        print(f"  Bear bias<{bear_bias}: 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# PART 3: Skip-month rules (crash protection)
# ================================================================
print("\n" + "=" * 60)
print("PART 3: SKIP-MONTH RULES")
print("=" * 60)

df_filtered = apply_filters(df.copy())

# Skip if market volatility > X
for vol_limit in [0.06, 0.08, 0.10, 0.12, 0.15]:
    ann_t, mdd_t, cal_t, nav_t, _, skipped = run_strategy(
        df_filtered, skip_vol=vol_limit)
    imp = ann_t - baseline_ann
    print(f"  Skip if vol>{vol_limit:.0%}: 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} "
          f"跳过{skipped}月 Δ{imp*100:+.1f}pp")

# Skip if market drawdown > X%
print()
for dd_limit in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    ann_t, mdd_t, cal_t, nav_t, _, skipped = run_strategy(
        df_filtered, skip_dd=dd_limit)
    imp = ann_t - baseline_ann
    print(f"  Skip if DD<-{dd_limit:.0%}: 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} "
          f"跳过{skipped}月 Δ{imp*100:+.1f}pp")

# ================================================================
# PART 4: Quality gate in bear markets
# ================================================================
print("\n" + "=" * 60)
print("PART 4: QUALITY GATE IN BEAR MARKETS")
print("=" * 60)

# In bear: only select stocks with 归母净利润_ttm > 0
df_filtered = apply_filters(df.copy())

# Modify run to add quality filter in bear
def run_with_quality(data, bear_quality=False):
    curves = []
    for date, grp in data.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]

        if regime == 'bull':
            tp, n_stocks = 0.30, 6
        else:
            tp, n_stocks = 0.22, 4
            if bear_quality:
                grp = grp[grp['归母净利润_ttm'] > 0]
                if len(grp) == 0:
                    curves.append((date, 0.0))
                    continue
                # Re-rank by market cap within quality stocks
                grp = grp.copy()
                grp['排名'] = grp['总市值'].rank()
                grp = grp[grp['排名'] <= n_stocks]

        daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]
        if len(daily_lists) == 0:
            curves.append((date, 0.0))
            continue

        final_rets = []
        for dl in daily_lists:
            modified, triggered = apply_tp(dl, tp)
            cr = np.prod([1 + r for r in modified])
            if not triggered: cr *= (1 - sell_cost)
            final_rets.append(cr)
        ret = np.mean(final_rets) * (1 - c_rate) - 1
        curves.append((date, ret))

    df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
    df_e['nav'] = (df_e['ret'] + 1).cumprod()
    days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
    ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
    df_e['max2here'] = df_e['nav'].expanding().max()
    df_e['dd'] = df_e['nav'] / df_e['max2here'] - 1
    mdd = df_e['dd'].min()
    calmar = ann / abs(mdd)
    return ann, mdd, calmar, df_e['nav'].iloc[-1]

ann_t, mdd_t, cal_t, nav_t = run_with_quality(df_filtered, bear_quality=True)
imp = ann_t - baseline_ann
print(f"Bear quality gate (净利润>0): 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} Δ{imp*100:+.1f}pp")

# Also test: quality gate in BOTH regimes
def run_with_quality_both(data):
    curves = []
    for date, grp in data.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]

        # Quality gate in both regimes
        grp = grp[grp['归母净利润_ttm'] > 0]
        if len(grp) == 0:
            curves.append((date, 0.0))
            continue

        grp = grp.copy()
        grp['排名'] = grp['总市值'].rank()

        if regime == 'bull':
            tp, n_stocks = 0.30, 6
        else:
            tp, n_stocks = 0.22, 4

        grp = grp[grp['排名'] <= n_stocks]
        daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

        if len(daily_lists) == 0:
            curves.append((date, 0.0))
            continue

        final_rets = []
        for dl in daily_lists:
            modified, triggered = apply_tp(dl, tp)
            cr = np.prod([1 + r for r in modified])
            if not triggered: cr *= (1 - sell_cost)
            final_rets.append(cr)
        ret = np.mean(final_rets) * (1 - c_rate) - 1
        curves.append((date, ret))

    df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
    df_e['nav'] = (df_e['ret'] + 1).cumprod()
    days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
    ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
    df_e['max2here'] = df_e['nav'].expanding().max()
    df_e['dd'] = df_e['nav'] / df_e['max2here'] - 1
    mdd = df_e['dd'].min()
    calmar = ann / abs(mdd)
    return ann, mdd, calmar, df_e['nav'].iloc[-1]

ann_t, mdd_t, cal_t, nav_t = run_with_quality_both(df_filtered)
imp = ann_t - baseline_ann
print(f"Both quality gate (净利润>0): 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} Δ{imp*100:+.1f}pp")


# ================================================================
# PART 5: Multi-timeframe bias signal
# ================================================================
print("\n" + "=" * 60)
print("PART 5: MULTI-TIMEFRAME BIAS SIGNAL")
print("=" * 60)

# Instead of just bias_20, test composite: bias_5 + bias_10 + bias_20
df_test = df.copy()
df_test = df_test[df_test['上市至今交易天数'] > 250]
df_test = df_test[~df_test['股票代码'].str.contains('bj')]

ind_val = df_test.groupby([ind_col, '交易日期']).agg(
    med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()
ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
df_test = df_test.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
df_test['val_pct'] = df_test['val_pct'].fillna(0.5)
df_test = df_test[df_test['val_pct'] < 0.68]

# Test composite bias: z-score of (bias_5 + bias_10 + bias_20)/3
df_test['bias_composite'] = (df_test['bias_5'] + df_test['bias_10'] + df_test['bias_20']) / 3

for bc in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
    d = df_test.copy()
    cutoff = d.groupby('交易日期')['bias_composite'].transform(lambda x: x.quantile(bc))
    d = d[d['bias_composite'] < cutoff]
    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) > 0:
        ann_t, mdd_t, cal_t, nav_t, _, _ = run_strategy(d)
        imp = ann_t - baseline_ann
        print(f"  bias_composite<{bc}: 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} Δ{imp*100:+.1f}pp")

# Test: bias_5 instead of bias_20
for bc in [0.45, 0.48, 0.50, 0.52, 0.55]:
    d = df_test.copy()
    cutoff = d.groupby('交易日期')['bias_5'].transform(lambda x: x.quantile(bc))
    d = d[d['bias_5'] < cutoff]
    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) > 0:
        ann_t, mdd_t, cal_t, nav_t, _, _ = run_strategy(d)
        imp = ann_t - baseline_ann
        print(f"  bias_5<{bc}: 年化{ann_t*100:.1f}% MDD{mdd_t*100:.1f}% Calmar{cal_t:.2f} Δ{imp*100:+.1f}pp")
