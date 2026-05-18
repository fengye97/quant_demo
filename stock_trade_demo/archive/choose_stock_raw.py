"""
邢不行-股票量化入门训练营
邢不行微信：xbx9585
Day3：选股策略示例
"""
# 导入必要的Python库
import ast  # 用于安全地评估字符串形式的Python表达式
import os  # 操作系统相关功能
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库

# 设置pandas显示选项，防止列太多时显示不清楚
pd.set_option('expand_frame_repr', False)

# ===选股参数设定
select_stock_num = 3  # 恢复为固定选股数量
c_rate = 1.2 / 10000  # 交易手续费率，万分之1.2
t_rate = 1 / 1000  # 印花税率，千分之一

# ===导入股票数据
# 从CSV文件读取所有股票数据，指定编码为gbk，解析交易日期列，不限制内存使用
# 数据包含每只股票的历史交易信息
df = pd.read_csv('供选股数据.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)  # 从csv文件中读取整理好的所有股票数据


# ==========只需要修改以下部分代码==========

# ===构建选股因子
# 这里使用总市值作为选股因子，值越小表示股票市值越小
# 你可以修改这里使用其他因子，可选因子从"所有可选因子在本文档第一行.csv"文件中选取
df['因子'] = df['总市值']
# df['因子'] = df['总市值'] - df['流通市值']

# ===对股票数据进行筛选
# 删除上市不满一年的股票(交易天数>250天)
df = df[df['上市至今交易天数'] > 250]  # 删除上市不满一年的股票
df = df[~df['股票代码'].str.contains('bj')] #去除北交所，~代表反选
# df = df[df['MACD'] <= 0]
# df = df[df['新版申万一级行业名称'].isin(['电子','汽车'])]  # 对所在行业进行筛选
# # '钢铁', '交通运输', '房地产', '公用事业', '化工', '休闲服务', '医药生物', '商业贸易', '食品饮料', '家用电器', '轻工制造', '纺织服装', '综合', '农林牧渔', '有色金属', '采掘', '电子', '银行', '汽车', '非银金融', '机械设备', '传媒', '国防军工', '建筑装饰', '通信', '电气设备', '计算机', '建筑材料'
# df = df[df['交易日期']>=pd.to_datetime('20210101')]  # 从21年开始
# df = df[df['交易日期']<=pd.to_datetime('20210101')]  # 从21年结束

# ===选股
df['排名'] = df.groupby('交易日期')['因子'].rank()  # 根据选股因子对股票进行排名
df = df[df['排名'] <= select_stock_num]  # 默认为最小的值获取最大的排名，

#如果想要把最大的值获取最大的排名，可以在因子前面加上-，比如df['因子'] = -df['总市值']


# ==========只需要修改以上部分代码==========



# ===整理选中股票数据，计算涨跌幅
# 挑选出选中股票
df['股票代码'] += ' '
df['股票名称'] += ' '
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))
group = df.groupby('交易日期')
select_stock = pd.DataFrame()
select_stock['买入股票代码'] = group['股票代码'].sum()
select_stock['买入股票名称'] = group['股票名称'].sum()

# 计算下周期每天的资金曲线
select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(lambda x: np.cumprod(np.array(list(x))+1, axis=1).mean(axis=0))
# 扣除买入手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
# 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])
# 计算下周期整体涨跌幅
select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
del select_stock['选股下周期每天资金曲线']


# 计算整体资金曲线
select_stock.reset_index(inplace=True)
select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()


def strategy_evaluate(select_stock):
    """
    :param select_stock: 每周期选出的股票
    :return:
    """
    results = pd.DataFrame()
    # ===计算累积净值
    results.loc[0, '累积净值'] = round(select_stock['累积净值'].iloc[-1], 2)
    # ===计算年化收益
    annual_return = (select_stock['累积净值'].iloc[-1]) ** (
            '1 days 00:00:00' / (select_stock['交易日期'].iloc[-1] - select_stock['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # 计算当日之前的资金曲线的最高点
    select_stock['max2here'] = select_stock['累积净值'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    select_stock['dd2here'] = select_stock['累积净值'] / select_stock['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(select_stock.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = select_stock[select_stock['交易日期'] <= end_date].sort_values(by='累积净值', ascending=False).iloc[0]['交易日期']
    # 将无关的变量删除
    select_stock.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤周期开始时间'] = str(start_date)
    results.loc[0, '最大回撤周期结束时间'] = str(end_date)
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)
    return results.T


# 计算累积净值
select_stock['累积净值'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()

# 评估策略表现
results = strategy_evaluate(select_stock)
print(results)


# 保存策略结果
select_stock.to_csv('选股策略详情.csv', encoding='gbk')

# ===画图
select_stock.set_index('交易日期', inplace=True)
plt.plot(select_stock['资金曲线'])
plt.show()
df.to_csv('xingbuxing_stock_data.csv', index=False) #导出选Gu表格