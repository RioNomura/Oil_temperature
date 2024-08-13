import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('AI-Engineer/multivariate-time-series-prediction/ett.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 基本統計量の確認
print(df.describe())

# 時系列のトレンドと季節性の確認
fig, ax = plt.subplots(figsize=(12, 6))
df['OT'].plot(ax=ax)
ax.set_title('Oil Temperature (OT) Time Series')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
plt.show()

# 季節性の確認
monthly_mean = df['OT'].groupby(df.index.month).mean()
fig, ax = plt.subplots(figsize=(8, 6))
monthly_mean.plot(ax=ax)
ax.set_title('Monthly Mean Oil Temperature')
ax.set_xlabel('Month')
ax.set_ylabel('Temperature (°C)')
plt.show()

# 異常値の確認
z_scores = ((df['OT'] - df['OT'].mean()) / df['OT'].std()).abs()
outliers = df[z_scores > 3]
print('Outliers:')
print(outliers)