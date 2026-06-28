"""
Explore unused columns and new factor ideas systematically
"""
import pandas as pd, numpy as np, ast

pd.set_option('expand_frame_repr', False)

# === Load ===
df = pd.read_csv('stock_data.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))

# Build regime
mkt_ret = df.groupby('交易日期')['涨跌幅'].mean()
mkt_cum = (1 + mkt_ret).cumprod()
mkt_ma12 = mkt_cum.rolling(12).mean()
regime_map = (mkt_cum > mkt_ma12).map({True: 'bull', False: 'bear'})
df['市场状态'] = df['交易日期'].map(regime_map)

# Baseline params
c_rate = 1.2 / 10000
sell_cost = c_rate + 1/1000
ind_col = '新版申万二级行业名称'
BULL_TP, BEAR_TP = 0.30, 0.22
BULL_N, BEAR_N = 6, 4

# === Benchmark: baseline strategy ===
def run_full_strategy(data, extra_filters=None):
    """Run baseline with optional extra filters"""
    d = data.copy()
    d = d[d['上市至今交易天数'] > 250]
    d = d[~d['股票代码'].str.contains('bj')]

    ind_val = d.groupby([ind_col, '交易日期']).agg(
        med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

    def calc_val_percentile(grp):
        grp = grp.sort_values('交易日期')
        ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
        bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
        grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
        return grp

    ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
    d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    d['val_pct'] = d['val_pct'].fillna(0.5)
    d = d[d['val_pct'] < 0.68]

    cutoff = d.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(0.52))
    d = d[d['bias_20'] < cutoff]

    # Extra filter before ranking
    if extra_filters:
        for col, op, val in extra_filters:
            if op == '>':
                d = d[d[col] > val]
            elif op == '<':
                d = d[d[col] < val]
            elif op == 'pct_lt':
                cutoff = d.groupby('交易日期')[col].transform(lambda x: x.quantile(val))
                d = d[d[col] < cutoff]
            elif op == 'pct_gt':
                cutoff = d.groupby('交易日期')[col].transform(lambda x: x.quantile(val))
                d = d[d[col] > cutoff]

    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) == 0:
        return 0, 0, 0, 0

    # Strategy execution
    curves = []
    for date, grp in d.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]
        tp, n_stocks = (BULL_TP, BULL_N) if regime == 'bull' else (BEAR_TP, BEAR_N)
        daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

        final_rets = []
        for dl in daily_lists:
            cumret = 1.0; result = []; triggered = False
            for r in dl:
                if triggered: result.append(0.0); continue
                cumret *= (1 + r); result.append(r)
                if cumret - 1 > tp: triggered = True; result[-1] = result[-1] - sell_cost
            cr = np.prod([1 + r_ for r_ in result])
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


ann_bl, mdd_bl, cal_bl, nav_bl = run_full_strategy(df.copy())
print(f"BASELINE: 年化{ann_bl*100:.1f}% MDD{mdd_bl*100:.1f}% Calmar{cal_bl:.2f} 净值{nav_bl:.0f}")
print()

# ================================================================
# TEST 1: Alternative market cap (流通市值 vs 总市值)
# ================================================================
print("=" * 60)
print("1. 流通市值 vs 总市值 as size factor")
print("=" * 60)

for cap_col, cap_name in [('总市值', '总市值'), ('流通市值', '流通市值')]:
    d = df.copy()
    d = d[d['上市至今交易天数'] > 250]
    d = d[~d['股票代码'].str.contains('bj')]

    ind_val = d.groupby([ind_col, '交易日期']).agg(
        med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

    def calc_val_percentile(grp):
        grp = grp.sort_values('交易日期')
        ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
        bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
        grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
        return grp

    ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
    d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    d['val_pct'] = d['val_pct'].fillna(0.5)
    d = d[d['val_pct'] < 0.68]

    cutoff = d.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(0.52))
    d = d[d['bias_20'] < cutoff]

    d['因子'] = d[cap_col]
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) > 0:
        curves = []
        for date, grp in d.groupby('交易日期'):
            regime = grp['市场状态'].iloc[0]
            tp, n_stocks = (BULL_TP, BULL_N) if regime == 'bull' else (BEAR_TP, BEAR_N)
            daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

            final_rets = []
            for dl in daily_lists:
                cumret = 1.0; result = []; triggered = False
                for r in dl:
                    if triggered: result.append(0.0); continue
                    cumret *= (1 + r); result.append(r)
                    if cumret - 1 > tp: triggered = True; result[-1] = result[-1] - sell_cost
                cr = np.prod([1 + r_ for r_ in result])
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
        imp = ann - ann_bl
        print(f"  {cap_name}: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{calmar:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 2: Earnings growth filter
# ================================================================
print("\n" + "=" * 60)
print("2. Earnings Growth Filter (归母净利润_ttm同比)")
print("=" * 60)

for growth_cut in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]:
    ann, mdd, cal, nav = run_full_strategy(df.copy(), [('归母净利润_ttm同比', '>', growth_cut)])
    imp = ann - ann_bl
    print(f"  净利润同比>{growth_cut*100:.0f}%: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 3: Cash flow quality
# ================================================================
print("\n" + "=" * 60)
print("3. Cash Flow Quality (经营活动现金流)")
print("=" * 60)

for cf_cut in [0, 1e8, 5e8, 1e9]:
    ann, mdd, cal, nav = run_full_strategy(df.copy(), [('经营活动产生的现金流量净额_ttm', '>', cf_cut)])
    imp = ann - ann_bl
    print(f"  经营现金流TTM>{cf_cut/1e8:.0f}亿: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 4: Profit quality (ROE proxy = 归母净利润_ttm / 净资产)
# ================================================================
print("\n" + "=" * 60)
print("4. ROE Filter (净利润TTM/净资产)")
print("=" * 60)

df_roe = df.copy()
df_roe['roe_ttm'] = df_roe['归母净利润_ttm'] / df_roe['净资产'].replace(0, np.nan)

for roe_cut in [-0.1, 0.0, 0.02, 0.05, 0.08, 0.10]:
    d = df_roe.copy()
    d = d[d['roe_ttm'].notna()]
    ann, mdd, cal, nav = run_full_strategy(d, [('roe_ttm', '>', roe_cut)])
    imp = ann - ann_bl
    print(f"  ROE>{roe_cut*100:.0f}%: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 5: Price position within month (close-low)/(high-low)
# ================================================================
print("\n" + "=" * 60)
print("5. Price Position (收盘价在月内位置)")
print("=" * 60)

df_pp = df.copy()
df_pp['price_position'] = (df_pp['收盘价'] - df_pp['最低价']) / (df_pp['最高价'] - df_pp['最低价'] + 0.01)

for pp_cut in [0.2, 0.3, 0.4, 0.5]:
    ann, mdd, cal, nav = run_full_strategy(df_pp, [('price_position', '<', pp_cut)])
    imp = ann - ann_bl
    print(f"  价格位置<{pp_cut}: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 6: Short-term reversal (bias_5)
# ================================================================
print("\n" + "=" * 60)
print("6. Short-term Reversal (bias_5)")
print("=" * 60)

for bias_col, bias_name in [('bias_5', 'bias_5'), ('bias_10', 'bias_10'), ('bias_20', 'bias_20')]:
    best_ann, best_cut = 0, 0
    for bc in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58]:
        d = df.copy()
        d = d[d['上市至今交易天数'] > 250]
        d = d[~d['股票代码'].str.contains('bj')]

        ind_val = d.groupby([ind_col, '交易日期']).agg(
            med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

        def calc_val_percentile(grp):
            grp = grp.sort_values('交易日期')
            ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
            bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
            grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
            return grp

        ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
        d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
        d['val_pct'] = d['val_pct'].fillna(0.5)
        d = d[d['val_pct'] < 0.68]

        cutoff = d.groupby('交易日期')[bias_col].transform(lambda x: x.quantile(bc))
        d = d[d[bias_col] < cutoff]

        d['因子'] = d['总市值']
        d['排名'] = d.groupby('交易日期')['因子'].rank()
        d = d[d['排名'] <= 6]

        if len(d) > 0:
            curves = []
            for date, grp in d.groupby('交易日期'):
                regime = grp['市场状态'].iloc[0]
                tp, n_stocks = (BULL_TP, BULL_N) if regime == 'bull' else (BEAR_TP, BEAR_N)
                daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

                final_rets = []
                for dl in daily_lists:
                    cumret = 1.0; result = []; triggered = False
                    for r in dl:
                        if triggered: result.append(0.0); continue
                        cumret *= (1 + r); result.append(r)
                        if cumret - 1 > tp: triggered = True; result[-1] = result[-1] - sell_cost
                    cr = np.prod([1 + r_ for r_ in result])
                    if not triggered: cr *= (1 - sell_cost)
                    final_rets.append(cr)
                ret = np.mean(final_rets) * (1 - c_rate) - 1
                curves.append((date, ret))

            df_e = pd.DataFrame(curves, columns=['交易日期', 'ret'])
            df_e['nav'] = (df_e['ret'] + 1).cumprod()
            days = (df_e['交易日期'].iloc[-1] - df_e['交易日期'].iloc[0]).days
            ann = df_e['nav'].iloc[-1] ** (365.0 / days) - 1
            if ann > best_ann: best_ann, best_cut = ann, bc

    imp = best_ann - ann_bl
    print(f"  {bias_name} 最优<{best_cut}: 年化{best_ann*100:.1f}% Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 7: Low volatility anomaly (涨跌幅std)
# ================================================================
print("\n" + "=" * 60)
print("7. Low Volatility Filter")
print("=" * 60)

for vol_col in ['涨跌幅std_5', '涨跌幅std_10', '涨跌幅std_20', '振幅_20']:
    best_ann, best_cut = 0, 0
    for vc in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ann, mdd, cal, nav = run_full_strategy(df.copy(), [(vol_col, 'pct_lt', vc)])
        if ann > best_ann: best_ann, best_cut = ann, vc
    imp = best_ann - ann_bl
    print(f"  {vol_col}<P{best_cut*100:.0f}: 年化{best_ann*100:.1f}% Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 8: Low turnover/volume anomaly (成交额std)
# ================================================================
print("\n" + "=" * 60)
print("8. Low Turnover/Volume Anomaly")
print("=" * 60)

for vol_col in ['成交额std_5', '成交额std_10', '成交额std_20']:
    best_ann, best_cut = 0, 0
    for vc in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ann, mdd, cal, nav = run_full_strategy(df.copy(), [(vol_col, 'pct_lt', vc)])
        if ann > best_ann: best_ann, best_cut = ann, vc
    imp = best_ann - ann_bl
    print(f"  {vol_col}<P{best_cut*100:.0f}: 年化{best_ann*100:.1f}% Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 9: Industry concentration limit
# ================================================================
print("\n" + "=" * 60)
print("9. Industry Concentration Limit")
print("=" * 60)

def run_with_industry_limit(data, max_per_ind=2, ind_level='新版申万二级行业名称'):
    d = data.copy()
    d = d[d['上市至今交易天数'] > 250]
    d = d[~d['股票代码'].str.contains('bj')]

    ind_val = d.groupby([ind_col, '交易日期']).agg(
        med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

    def calc_val_percentile(grp):
        grp = grp.sort_values('交易日期')
        ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
        bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
        grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
        return grp

    ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
    d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    d['val_pct'] = d['val_pct'].fillna(0.5)
    d = d[d['val_pct'] < 0.68]

    cutoff = d.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(0.52))
    d = d[d['bias_20'] < cutoff]

    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()

    # Apply industry limit: per date, per industry, only keep top N by rank
    d = d.sort_values(['交易日期', '排名'])
    d['ind_rank'] = d.groupby(['交易日期', ind_level])['排名'].rank()
    d = d[d['ind_rank'] <= max_per_ind]
    # Then re-rank and take top 6
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) == 0:
        return 0, 0, 0, 0

    curves = []
    for date, grp in d.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]
        tp, n_stocks = (BULL_TP, BULL_N) if regime == 'bull' else (BEAR_TP, BEAR_N)
        daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

        final_rets = []
        for dl in daily_lists:
            cumret = 1.0; result = []; triggered = False
            for r in dl:
                if triggered: result.append(0.0); continue
                cumret *= (1 + r); result.append(r)
                if cumret - 1 > tp: triggered = True; result[-1] = result[-1] - sell_cost
            cr = np.prod([1 + r_ for r_ in result])
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

for max_per in [1, 2, 3]:
    ann, mdd, cal, nav = run_with_industry_limit(df.copy(), max_per)
    imp = ann - ann_bl
    print(f"  每行业最多{max_per}只: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 10: MACD filter (only buy when MACD > 0)
# ================================================================
print("\n" + "=" * 60)
print("10. MACD Direction Filter")
print("=" * 60)

ann, mdd, cal, nav = run_full_strategy(df.copy(), [('MACD', '>', 0)])
imp = ann - ann_bl
print(f"  MACD>0: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

ann, mdd, cal, nav = run_full_strategy(df.copy(), [('DIF', '>', 'DEA')])  # This might error, let's check
# Actually can't compare DIF>DEA with this filter system, skip

# ================================================================
# TEST 11: 涨跌幅_20 (1-month momentum) as additional filter
# ================================================================
print("\n" + "=" * 60)
print("11. Medium-term Momentum/Reversal")
print("=" * 60)

for mom_cut in [-0.15, -0.10, -0.05, 0.0]:
    ann, mdd, cal, nav = run_full_strategy(df.copy(), [('涨跌幅_20', '<', mom_cut)])
    imp = ann - ann_bl
    print(f"  涨跌幅_20<{mom_cut*100:.0f}%: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

for mom_cut in [0.05, 0.10, 0.15]:
    ann, mdd, cal, nav = run_full_strategy(df.copy(), [('涨跌幅_20', '>', mom_cut)])
    imp = ann - ann_bl
    print(f"  涨跌幅_20>{mom_cut*100:.0f}%: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 12: Composite Quality Score
# ================================================================
print("\n" + "=" * 60)
print("12. Composite Quality + Value Score")
print("=" * 60)

# Combine: EP, BP, profitability, low vol
df_comp = df.copy()
df_comp['ep_rank'] = df_comp.groupby('交易日期')['市盈率倒数'].rank(pct=True)
df_comp['bp_rank'] = df_comp.groupby('交易日期')['市净率倒数'].rank(pct=True)
df_comp['roe_raw'] = df_comp['归母净利润_ttm'] / df_comp['净资产'].replace(0, np.nan)
df_comp['roe_rank'] = df_comp.groupby('交易日期')['roe_raw'].rank(pct=True, na_option='bottom')
df_comp['vol_inv_rank'] = df_comp.groupby('交易日期')['涨跌幅std_20'].rank(pct=True, ascending=False)  # inverse vol

# Composite: average of value + quality ranks
df_comp['quality_score'] = df_comp[['ep_rank', 'bp_rank', 'roe_rank', 'vol_inv_rank']].mean(axis=1)

for q_cut in [0.3, 0.4, 0.5, 0.6]:
    ann, mdd, cal, nav = run_full_strategy(df_comp, [('quality_score', 'pct_gt', q_cut)])
    imp = ann - ann_bl
    print(f"  quality_score>P{q_cut*100:.0f}: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 13: VWAP deviation ((close-VWAP)/VWAP)
# ================================================================
print("\n" + "=" * 60)
print("13. VWAP Deviation Signal")
print("=" * 60)

df_vwap = df.copy()
df_vwap['vwap_dev'] = (df_vwap['收盘价'] - df_vwap['VWAP']) / df_vwap['VWAP']

for vd_cut in [-0.02, -0.01, 0, 0.01, 0.02]:
    ann, mdd, cal, nav = run_full_strategy(df_vwap, [('vwap_dev', '<', vd_cut)])
    imp = ann - ann_bl
    print(f"  VWAP偏离<{vd_cut*100:.0f}%: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# TEST 14: Regime-specific quality (narrow loss years)
# ================================================================
print("\n" + "=" * 60)
print("14. Bear-only Quality Gate")
print("=" * 60)

def run_bear_quality(data, bear_cf_cut=0):
    """Only apply quality filter in bear markets"""
    d = data.copy()
    d = d[d['上市至今交易天数'] > 250]
    d = d[~d['股票代码'].str.contains('bj')]

    ind_val = d.groupby([ind_col, '交易日期']).agg(
        med_ep=('市盈率倒数', 'median'), med_bp=('市净率倒数', 'median')).reset_index()

    def calc_val_percentile(grp):
        grp = grp.sort_values('交易日期')
        ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
        bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
        grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
        return grp

    ind_val = ind_val.groupby(ind_col, group_keys=False).apply(calc_val_percentile)
    d = d.merge(ind_val[[ind_col, '交易日期', 'val_pct']], on=[ind_col, '交易日期'], how='left')
    d['val_pct'] = d['val_pct'].fillna(0.5)
    d = d[d['val_pct'] < 0.68]

    cutoff = d.groupby('交易日期')['bias_20'].transform(lambda x: x.quantile(0.52))
    d = d[d['bias_20'] < cutoff]

    # Bear-only quality: remove loss-making companies in bear markets
    bear_mask = d['市场状态'] == 'bear'
    d = d[~(bear_mask & (d['归母净利润_ttm'] < bear_cf_cut))]

    d['因子'] = d['总市值']
    d['排名'] = d.groupby('交易日期')['因子'].rank()
    d = d[d['排名'] <= 6]

    if len(d) == 0:
        return 0, 0, 0, 0

    curves = []
    for date, grp in d.groupby('交易日期'):
        regime = grp['市场状态'].iloc[0]
        tp, n_stocks = (BULL_TP, BULL_N) if regime == 'bull' else (BEAR_TP, BEAR_N)
        daily_lists = list(grp['下周期每天涨跌幅'])[:n_stocks]

        final_rets = []
        for dl in daily_lists:
            cumret = 1.0; result = []; triggered = False
            for r in dl:
                if triggered: result.append(0.0); continue
                cumret *= (1 + r); result.append(r)
                if cumret - 1 > tp: triggered = True; result[-1] = result[-1] - sell_cost
            cr = np.prod([1 + r_ for r_ in result])
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

ann, mdd, cal, nav = run_bear_quality(df.copy(), 0)
imp = ann - ann_bl
print(f"  Bear剔除亏损企业: 年化{ann*100:.1f}% MDD{mdd*100:.1f}% Calmar{cal:.2f} Δ{imp*100:+.1f}pp")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY: All tests vs baseline 年化{:.1f}%".format(ann_bl*100))
print("=" * 60)
