"""
邢不行-股票量化入门训练营
邢不行微信：xbx9585
Day4：择时策略示例
"""
import pandas as pd
import matplotlib.pyplot as plt


# ===导入数据
df = pd.read_csv('创造191_历史选股记录_周_30.csv', 
                 encoding='GBK', 
                 parse_dates=['卖出日期'])


# ===计算择时资金曲线
df['择时资金曲线'] = (df['持仓涨跌'] * df['pos'] + 1).cumprod()
print('资金曲线：', df['资金曲线'].iloc[-1])
print('择时资金曲线：', df['择时资金曲线'].iloc[-1])


# ===计算最大回撤：邢不行公众号文章《如何通过3行Python代码计算最大回撤》
for var in ['资金曲线', '择时资金曲线']:
    # 计算当日之前的资金曲线的最高点
    df['max2here'] = df[var].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    df['dd2here'] = df[var] / df['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    max_draw_down = df.sort_values(by=['dd2here']).iloc[0]['dd2here']
    print(var, '最大回撤：', max_draw_down)


# ===画图
df.set_index('卖出日期', inplace=True)
plt.plot(df['资金曲线'])
plt.plot(df['择时资金曲线'])
plt.show()
