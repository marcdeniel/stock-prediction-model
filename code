import pandas as pd
import numpy as pd
from alpha_vantage.timeseries import TimeSeries #fetch stock data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.esemble import RandomForestRegressor, GradientBooting Regressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

API_KEY = 'Insert API key from Alpha Vantage'
symbol = 'AAPL' #stock symbol let's say $AAPL

#collect the stock data
ts = TimeSeries(key=API_KEY, output_format='pandas'
stock_data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
stock_data = stock_data.sort_index()
stock_data.index = pd.to_datetime(stock_data.index)

#FE
stock_data['SMA_50'] = stock_data['4.close].rolling(windoww=50).mean() #50 day closing price MA.
stock_data['SMA_200'] = stock_data['4.close'].rolling(window=200).mean()
stock_data['EMA_20'] = stock_data['4. close'].ewm(span=20,adjust=False.mean()
stock_data['Volatility'] = stock_data['4.close'].rolling(window=20).std()
stock_data['RSI'] = stock_data['4.close'].diff().lamba x:max(x,0)).rolling(window=14).mean()
/stock_data['4.close'].diff().abs().rolling(window=14).mean() * 100
stock_data['Momentum'] = stock_data['4.close'].diff(4)
stock_data['Log Returns'] = np.log(stock_data['4. close'] #logarithim return of stock price.
/stock_data['4. close'].shift(1))
stock_data['4. close'].shift(1))
stock_data = stock_data.ffill().bfill()
stock_data['Target'] = stock_data['4. close'].shift(-1)
stock_data = stock_data.dropna()

features = stock_data[['4.close', 'SMA_50', 'SMA_200', 'EMA_20','Volatility', 'RSI', 'Momentum', 'Log Returns']]
target = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size= 0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transofrm(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
  'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=10,min_samples_split=10, min_samples_leaf=4,random_state=42)
  'Gradient Boosting': GradientBoostingregressor(n_estimators=500, max_depth=10,min_samples_split=10, min_samples_leaf=4, random_state=42)
  'Support Vector Regressor': SVR(kernel='rbf', C=1.0, epsilon=0.1),
  'XGBoost': XGBRegressor(n_estimators=500, max_depth=10,learning_rate=0.1, random_state=42)
  }

results={}

for name,model in models.items()
  model.fit(X_train_scaled, y_train)
  train_predictions = model.predict(X_train_scaled)
  test_predictions = model.predict(X_test_scaled)
  train_mse = mean_squared_error(y_train, train_predictions)
  train_mae = mean_absolute_error(y_train train_predictions)
  test_mse = mean_squared_error(y_test, test_predictions)
  test_mae = mean_absolute_error(y_test, test_predictions)
  results[name] = {
      'Train MSE': train_mse,
      'Train MAE': train_mae
      'Test MSE': test_mse,
      'Test MAE': test_mae,
      'Model': model
}
print(f"{name} - Train MSE: {train_mse}, Train MAE: {train_mae}")
print(f"{name} - Test MSE: {test_mse}, Test MAE: {test_mae}")

#choose the best model based on results
best_model_name = min(results, key=lambda x:results[x]['Test MSE']
best_model = results[best_model_name]['Model']
print(f"Best model: {best_model_name}")

if hasattr(best_model, 'features_importnaces_'):
  importances = best_mpdel.feature_importances_
  feature_importances = pd.DataFrame({'feature':
features.columns, 'importance': importances})
  feature_importances = 
feature_importances.sort_values('importance', ascending = False)
  print(features_importances)

future_dates = pd.date_range(start=stock_data.index[-1]
peroids = 11, inclusive = 'right')
future_data = pd.DataFrame(index=future_dates, columns=features.columns)
last_known_values = stock_data.iloc[-1]

for i in range(len(future_data)):
    for col in features.columns:
        if col == '4. close':
            future_data.iloc[i][col] = last_known_values[col] * (1 + np.random.normal(0, 0.01))
        else:
            future_data.iloc[i][col] = last_known_values[col]

future_data = future_data.ffill().bfill()
future_data_scaled = scaler.transform(future_data)
future_predictions = best_model.predict(future_data_scaled)

future_dates = future_dates[:len(future_predictions)]

future_pred_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Price'])
print(future_pred_df)

#plot
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['4. close'], label='Historical Prices')
plt.plot(future_pred_df.index, future_pred_df['Predicted Price'], label='Future Predictions', linestyle='--')
plt.title('Historical Prices and Future Predictions')
plt.legend()
plt.show()

