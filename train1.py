import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

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
    df[f'{col}_lag6'] = df[col].shift(6)
    df[f'{col}_lag12'] = df[col].shift(12)
    df[f'{col}_lag24'] = df[col].shift(24)

df = df.dropna()

# スケーリング
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# データの分割
X = df_scaled.drop('OT', axis=1)
y = df_scaled['OT']

# 特徴量選択（初期段階）
selector = SelectKBest(score_func=f_regression, k=30)
X_selected = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()], index=X.index)

# Random Forestを使用して特徴量の重要度を計算
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_selected, y)

# 特徴量の重要度を取得
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X_selected.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

# 上位n個の特徴量を選択
n = 5  # 選択したい特徴量の数
top_features = feature_importances['feature'][:n].tolist()

# 選択された特徴量のみを使用
X_selected = X_selected[top_features]

# 時系列交差検証
tscv = TimeSeriesSplit(n_splits=5)

# モデルのリスト
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
}

# 評価指標の計算
def calculate_metrics(y_true, y_pred):
    return {
        'R-squared': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# モデルの評価
results = {}

for name, model in models.items():
    train_scores = []
    test_scores = []
    
    for train_index, test_index in tscv.split(X_selected):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_scores.append(calculate_metrics(y_train, y_train_pred))
        test_scores.append(calculate_metrics(y_test, y_test_pred))
    
    results[name] = {
        'train': pd.DataFrame(train_scores).mean(),
        'test': pd.DataFrame(test_scores).mean()
    }

# ARIMAモデルの評価
arima_train_scores = []
arima_test_scores = []

for train_index, test_index in tscv.split(X_selected):
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = ARIMA(y_train, order=(1,1,1))
    fitted_model = model.fit()
    y_train_pred = fitted_model.fittedvalues
    y_test_pred = fitted_model.forecast(steps=len(y_test))
    
    arima_train_scores.append(calculate_metrics(y_train[1:], y_train_pred[1:]))  # ARIMAは1つ少ないデータポイントを返す
    arima_test_scores.append(calculate_metrics(y_test, y_test_pred))

results['ARIMA'] = {
    'train': pd.DataFrame(arima_train_scores).mean(),
    'test': pd.DataFrame(arima_test_scores).mean()
}

# 結果の出力を表形式に変換
train_df = pd.DataFrame({name: results[name]['train'] for name in results})
test_df = pd.DataFrame({name: results[name]['test'] for name in results})

# 有効数字5桁にフォーマット
train_df = train_df.applymap(lambda x: f'{x:.5g}')
test_df = test_df.applymap(lambda x: f'{x:.5g}')

# 表を画像として出力
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Train scores plot
axes[0].axis('off')
train_table = axes[0].table(cellText=train_df.values, colLabels=train_df.columns, rowLabels=train_df.index, loc='center')
train_table.auto_set_font_size(False)
train_table.set_fontsize(11)

# タイトルの位置を調整する
axes[0].set_title('Train Scores', pad=10)

# Test scores plot
axes[1].axis('off')
test_table = axes[1].table(cellText=test_df.values, colLabels=test_df.columns, rowLabels=test_df.index, loc='center')
test_table.auto_set_font_size(False)
test_table.set_fontsize(11)

axes[1].set_title('Test Scores', pad=10)

# 選択された特徴量を表示
print("特徴量一覧:")
print(top_features)

# 各モデルの特徴量重要度を表示（ツリーベースのモデルのみ）
for name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
    if name in models:
        importance = pd.DataFrame({
            'feature': X_selected.columns,
            'importance': models[name].feature_importances_
        }).sort_values('importance', ascending=False)
        print(f'\n{name} 特徴量重要度:')
        print(importance)

# 距離を調整する
plt.subplots_adjust(hspace=0.03)
plt.show()