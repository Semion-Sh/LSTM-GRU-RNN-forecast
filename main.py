import csv
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

url = 'https://api.binance.com/api/v3/klines'
# base_ticker = 'https://fapi.binance.com/fapi/v1/ticker/price'
# base_ticker_24 = 'https://api.binance.com/api/v3/ticker/24hr'

params = {
    'symbol': 'BTCUSDT',
    'interval': '1d',
    'limit': 1000
}

# получаем данные и сохраняем в CSV файл
response = requests.get(url, params=params)
data = response.json()
header = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
          'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
df = pd.DataFrame(data, columns=header)
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
df.to_csv('binance_crypto_quotes.csv', index=False)

# Загрузка данных из CSV-файла в Pandas DataFrame
df = pd.read_csv('binance_crypto_quotes.csv')

# Предварительная обработка данных
df.dropna(inplace=True)
df['Open time'] = pd.to_datetime(df['Open time'])
df.set_index('Open time', inplace=True)
df['Close'] = df['Close'].astype('float64')

# Масштабирование данных
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(df_scaled) * 0.8)
train_data = df_scaled.iloc[:train_size]
test_data = df_scaled.iloc[train_size:]


# Создание функции для создания обучающих и тестовых данных в формате временных рядов
def create_time_series_data(dataset, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back):
        data_X.append(dataset[i:(i + look_back), 0])
        data_Y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_Y)


# Создание обучающих и тестовых данных в формате временных рядов с окном 60 дней
look_back = 60
train_X, train_Y = create_time_series_data(train_data.values, look_back)
test_X, test_Y = create_time_series_data(test_data.values, look_back)

# Изменение размерности данных для LSTM-модели
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Создание и обучение LSTM-модели
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train_X, train_Y, epochs=100, batch_size=32)

# оценка качества модели на тестовой выборке
test_loss = model.evaluate(test_X, test_Y)
print('Средняя абсолютная ошибка (MAE) на тестовой выборке:', test_loss)

# получение предсказаний модели на тестовой выборке
y_pred = model.predict(test_X)

# подсчет MSE
mse = mean_squared_error(test_Y, y_pred)
print('Среднеквадратичная ошибка (MSE) на тестовой выборке:', mse)

# получение предсказаний модели на тестовой выборке
y_pred = model.predict(test_X)

plt.plot(test_Y, label='Данные')
plt.plot(y_pred, label='Предсказание')
plt.plot(len(test_Y) + 1, model.predict(np.reshape(test_X[-1], (1, look_back, 1))), 'ro',
         label='Предсказание на 1 час вперед')
plt.legend()
plt.show()
