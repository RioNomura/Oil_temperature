import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# データの読み込みと前処理
df = pd.read_csv('AI-Engineer/multivariate-time-series-prediction/ett.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.fillna(method='ffill')

# 特徴量エンジニアリング
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['weekday'] = df.index.dayofweek

# ラグ特徴量の追加
for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag12'] = df[col].shift(12)
    df[f'{col}_lag24'] = df[col].shift(24)

df = df.dropna()

# スケーリング
scaler = StandardScaler()

# データの分割
X = df.drop('OT', axis=1)
y = df['OT']

# 時系列分割
tscv = TimeSeriesSplit(n_splits=5)

# SARIMAXモデルの設定
sarimax_order = (1, 0, 1)  # p, d, qの順を単純化
sarimax_seasonal_order = (1, 1, 0, 12)  # P, D, Q, sの順を再設定

train_scores = []
test_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # スケーリング
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # SARIMAXモデルの訓練
    sarimax_model = SARIMAX(y_train_scaled, order=sarimax_order, seasonal_order=sarimax_seasonal_order, exog=X_train_scaled)
    fitted_sarimax = sarimax_model.fit(disp=False)

    # 予測
    y_pred_train = fitted_sarimax.fittedvalues
    y_pred_test = fitted_sarimax.forecast(steps=len(y_test_scaled), exog=X_test_scaled)

    # 評価
    train_scores.append({
        'R-squared': r2_score(y_train_scaled, y_pred_train),
        'MSE': mean_squared_error(y_train_scaled, y_pred_train),
        'RMSE': np.sqrt(mean_squared_error(y_train_scaled, y_pred_train)),
        'MAE': mean_absolute_error(y_train_scaled, y_pred_train)
    })
    test_scores.append({
        'R-squared': r2_score(y_train_scaled, y_pred_train),
        'MSE': mean_squared_error(y_test_scaled, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test_scaled, y_pred_test)),
        'MAE': mean_absolute_error(y_test_scaled, y_pred_test)
    })

# 結果の表示
train_df = pd.DataFrame(train_scores).mean()
test_df = pd.DataFrame(test_scores).mean()

print("Train Scores:")
print(train_df)
print("\nTest Scores:")
print(test_df)

# 有効数字5桁にフォーマット
train_df = train_df.apply(lambda x: f'{x:.5g}')
test_df = test_df.apply(lambda x: f'{x:.5g}')

# 表を画像として出力
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Train scores plot
axes[0].axis('off')
train_table = axes[0].table(cellText=[train_df.values], colLabels=train_df.index, loc='center')
train_table.auto_set_font_size(False)
train_table.set_fontsize(11)
axes[0].set_title('Train Scores', pad=3, loc='center')

# Test scores plot
axes[1].axis('off')
test_table = axes[1].table(cellText=[test_df.values], colLabels=test_df.index, loc='center')
test_table.auto_set_font_size(False)
test_table.set_fontsize(11)
axes[1].set_title('Test Scores', loc='center')

plt.subplots_adjust(hspace=0.03)
plt.show()
