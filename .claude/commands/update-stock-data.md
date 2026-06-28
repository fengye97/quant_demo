# Update Stock Data

更新 `stock_trade_demo/stock_data.csv` 中的 A 股月度数据到最新交易日。

## 执行步骤

1. **检查当前数据状态**：确认 CSV 中的最新日期，判断是否需要更新。

2. **清理旧数据和缓存**（如果目标月份已有数据）：
   - 删除 `.cache/daily_YYYY-MM.pkl` 和 `.cache/rtquotes_YYYY-MM.pkl`
   - 从 CSV 中移除目标月份的所有行
   - 重置上月的 `下周期每天涨跌幅` 字段为 `[]`

   使用以下 Python 代码：
   ```python
   python3 -c "
   import csv
   csv_path = 'stock_trade_demo/stock_data.csv'
   target_prefix = 'YYYY-MM'  # e.g. '2026-05'
   prev_prefix = 'YYYY-MM'    # e.g. '2026-04'

   with open(csv_path, 'r', encoding='gbk') as f:
       reader = csv.reader(f)
       headers = next(reader)
       rows = [row for row in reader]

   rows = [row for row in rows if not row[0].startswith(target_prefix)]
   for row in rows:
       if row[0].startswith(prev_prefix) and len(row) > 54:
           row[54] = '[]'

   with open(csv_path, 'w', encoding='gbk', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(headers)
       writer.writerows(rows)
   "
   ```

3. **运行数据补充脚本**（从 repo 根目录执行）：
   ```bash
   python3 get_stock_info.py --mode supplement --year YYYY --month M --cache-dir .cache
   ```

4. **验证结果**：
   ```python
   python3 -c "
   import csv
   dates = set()
   with open('stock_trade_demo/stock_data.csv', 'r', encoding='gbk') as f:
       reader = csv.reader(f)
       next(reader)
       for row in reader:
           if row[0].startswith('YYYY-MM'):
               dates.add(row[0])
   print(f'Dates: {sorted(dates)}')
   print(f'Latest: {max(dates)}')
   "
   ```

## 关键信息

- **数据源**：Sina Finance API（日K线）+ Tencent Finance API（实时行情/市值/PE/PB）
- **运行时间**：约 2-3 分钟（5000+ 股票，20 并发线程）
- **CSV 编码**：GBK
- **缓存位置**：`.cache/daily_YYYY-MM.pkl`（日K数据）, `.cache/rtquotes_YYYY-MM.pkl`（实时行情）
- **核心脚本**：`/Users/fatcat/Desktop/quant/get_stock_info.py`

## 数据结构

每只股票每月一行，包含：
- 月度 OHLCV + VWAP
- 流通/总市值（从上月延续）
- 财务数据（从上月延续：归母净利润、经营现金流、净资产等）
- 技术指标（从日线数据计算：bias、振幅、涨跌幅std、成交额std、KDJ、MACD）
- 市盈率/市净率倒数（从实时行情获取）
- 申万行业分类（从上月延续）
- 月涨跌幅 + 下周期每天涨跌幅

## 注意事项

- 如果目标月还没结束（月中更新），"date" 字段为当月最后一个交易日
- `下周期每天涨跌幅` 只有在下个月数据补充时才会被回填
- 如果同月内多次更新，需要先清理旧的月数据和缓存再重新运行
- 非交易日（周末/节假日）运行时拿到的仍是上一个交易日的数据
