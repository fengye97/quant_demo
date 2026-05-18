"""
convert_data.py — 将 stock_data.csv (GBK, 823MB) 转换为 Parquet 格式。

一次性运行即可：
    python3 convert_data.py

输出：
    stock_data.parquet — Snappy 压缩的 Parquet 文件（已加入 .gitignore）
"""

import os
import time
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, 'stock_data.csv')
PARQUET_PATH = os.path.join(DATA_DIR, 'stock_data.parquet')


def format_size(bytes_val):
    """以人性化单位格式化文件大小。"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def main():
    print("=" * 60)
    print("  stock_data.csv → Parquet 转换工具")
    print("=" * 60)

    # ── 1. 检查 CSV 是否存在 ──
    if not os.path.exists(CSV_PATH):
        print(f"\n错误: 未找到 CSV 文件: {CSV_PATH}")
        return 1

    csv_size = os.path.getsize(CSV_PATH)
    print(f"\n  CSV 路径:     {CSV_PATH}")
    print(f"  当前大小:      {format_size(csv_size)}")

    # ── 2. 读取 CSV ──
    print(f"\n读取 CSV (GBK 编码, 约 70 万行 × 55 列)...")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH, encoding='gbk',
                     parse_dates=['交易日期'], low_memory=False)
    elapsed = time.time() - t0
    print(f"  读取完成:      {df.shape[0]:,} 行 × {df.shape[1]} 列 ({elapsed:.1f}s)")

    # ── 3. 优化数据类型（缩小内存占用） ──
    print("\n优化数据类型...")
    # 交易日期已在上面解析为 datetime64
    # 数值列转为对应的 numpy 类型以便 Parquet 高效存储
    numeric_cols = ['总市值', 'bias_20', '成交额std_10', '市盈率倒数', '市净率倒数',
                    '最高价', '最低价', '收盘价', 'MACD', 'DIF', 'DEA',
                    '涨跌幅_20', '涨跌幅std_20', '成交额', '涨跌幅', '流通市值',
                    '开盘价', 'VWAP', '上市至今交易天数',
                    '涨跌幅_10', 'bias_5', 'bias_10', '振幅_5', '振幅_10', '振幅_20',
                    '涨跌幅std_5', '涨跌幅std_10', '成交额std_5', '成交额std_20',
                    'K', 'D', 'J',
                    '归母净利润', '归母净利润_ttm', '归母净利润_ttm同比',
                    '归母净利润_单季', '归母净利润_单季同比', '归母净利润_单季环比',
                    '经营活动产生的现金流量净额', '经营活动产生的现金流量净额_ttm',
                    '经营活动产生的现金流量净额_ttm同比',
                    '经营活动产生的现金流量净额_单季',
                    '经营活动产生的现金流量净额_单季同比',
                    '经营活动产生的现金流量净额_单季环比',
                    '净资产']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 字符串列保持为 object（Parquet 会自动处理为字符串）
    # 注意：下周期每天涨跌幅 是字符串形式的 list，保持不变

    before_mem = df.memory_usage(deep=True).sum()
    print(f"  DataFrame 内存: {format_size(before_mem)}")

    # ── 4. 写入 Parquet ──
    print(f"\n写入 Parquet (Snappy 压缩)...")
    t0 = time.time()
    df.to_parquet(PARQUET_PATH, engine='pyarrow', compression='snappy', index=False)
    elapsed = time.time() - t0
    parquet_size = os.path.getsize(PARQUET_PATH)

    print(f"  写入完成:      ({elapsed:.1f}s)")
    print(f"  Parquet 大小:   {format_size(parquet_size)}")

    # ── 5. 压缩率报告 ──
    ratio = csv_size / parquet_size
    savings = (1 - parquet_size / csv_size) * 100
    print(f"\n" + "=" * 60)
    print(f"  压缩率报告")
    print(f"  CSV 大小:      {format_size(csv_size)}")
    print(f"  Parquet 大小:   {format_size(parquet_size)}")
    print(f"  压缩比:         {ratio:.2f}x")
    print(f"  空间节省:       {savings:.1f}% ({format_size(csv_size - parquet_size)})")
    print(f"=" * 60)

    # ── 6. 数据完整性验证 ──
    print(f"\n数据完整性验证...")

    # 6a. 行数一致
    csv_rows = df.shape[0]
    parquet_rows = pq.read_metadata(PARQUET_PATH).num_rows
    assert csv_rows == parquet_rows, (
        f"行数不一致: CSV={csv_rows}, Parquet={parquet_rows}"
    )
    print(f"  行数一致:      {csv_rows:,} ✓")

    # 6b. 列数一致
    csv_cols = set(df.columns)
    parquet_cols = set(pq.read_schema(PARQUET_PATH).names)
    assert csv_cols == parquet_cols, (
        f"列不一致: 仅CSV有={csv_cols - parquet_cols}, "
        f"仅Parquet有={parquet_cols - csv_cols}"
    )
    print(f"  列数一致:      {len(csv_cols)} 列 ✓")

    # 6c. 随机抽样比较
    print(f"  随机样本比较...")
    n_samples = min(1000, csv_rows)
    sample_indices = df.sample(n=n_samples, random_state=42).index.sort_values()

    df_sample = df.loc[sample_indices].reset_index(drop=True)
    pq_df = pd.read_parquet(PARQUET_PATH)
    pq_sample = pq_df.loc[sample_indices].reset_index(drop=True)

    # 比较每列（Parquet 的 datetime 需要统一时区，这里做 safe compare）
    mismatches = 0
    for col in csv_cols:
        csv_vals = df_sample[col]
        pq_vals = pq_sample[col]

        if pd.api.types.is_datetime64_any_dtype(csv_vals) and pd.api.types.is_datetime64_any_dtype(pq_vals):
            # datetime 直接比较
            if not csv_vals.equals(pq_vals):
                mismatches += 1
        elif pd.api.types.is_float_dtype(csv_vals) or pd.api.types.is_float_dtype(pq_vals):
            # 浮点数允许一定精度误差
            csv_numeric = pd.to_numeric(csv_vals, errors='coerce').fillna(0)
            pq_numeric = pd.to_numeric(pq_vals, errors='coerce').fillna(0)
            if not np.allclose(csv_numeric, pq_numeric, rtol=1e-10, equal_nan=True):
                mismatches += 1
        else:
            # 字符串等直接比较
            # Parquet 读取回来的 datetime 会带 ns 精度，CSV 的 datetime 也是 ns
            # 但 object 类型列中可能混有 NaN
            if not csv_vals.fillna('__NA__').equals(pq_vals.fillna('__NA__')):
                mismatches += 1

    if mismatches == 0:
        print(f"  随机 {n_samples} 行 (共 {len(csv_cols)} 列) 完全一致 ✓")
    else:
        print(f"  发现 {mismatches} 列有差异 ✗")

    # ── 7. 读取性能报告 ──
    print(f"\n读取性能对比 (3 次平均):")

    # CSV 读取速度
    csv_times = []
    for _ in range(3):
        t0 = time.time()
        _ = pd.read_csv(CSV_PATH, encoding='gbk',
                        parse_dates=['交易日期'], low_memory=False)
        csv_times.append(time.time() - t0)
    avg_csv_time = sum(csv_times) / len(csv_times)

    # Parquet 读取速度
    pq_times = []
    for _ in range(3):
        t0 = time.time()
        _ = pd.read_parquet(PARQUET_PATH)
        pq_times.append(time.time() - t0)
    avg_pq_time = sum(pq_times) / len(pq_times)

    speedup = avg_csv_time / avg_pq_time
    print(f"  CSV 读取:       {avg_csv_time:.2f}s")
    print(f"  Parquet 读取:   {avg_pq_time:.2f}s")
    print(f"  读取加速:       {speedup:.1f}x")

    print(f"\n✓ 转换完成！Parquet 文件已生成: {PARQUET_PATH}")
    return 0


if __name__ == '__main__':
    import numpy as np
    exit(main())
