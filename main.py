from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService
from services.MarketDataService import MarketDataService

from config import SANDBOX_TOKEN

from datetime import timedelta
from utils.time import now, prepare_date

TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)

instrumentsService = InstrumentsService(TService)
marketDataService = MarketDataService(TService)


# response = instrumentsService.bonds()


response = instrumentsService.find_instrument('RU000A107UL4')
tbank_instrument = response.get('instruments')[0]



tbank_candles = marketDataService.get_candles(
    from_date=prepare_date(now() - timedelta(days=1)),
    instrumentId=tbank_instrument.get('figi'),
    limit=1,
)

print(tbank_candles)


import pandas as pd
def parse_quotation(quotation):
    """Преобразует units и nano в float."""
    return float(quotation['units']) + float(quotation['nano']) / 1e9



# Преобразуем данные
candles = tbank_candles['candles']
data = []

for candle in candles:
    data.append({
        'time': candle['time'],
        'open': parse_quotation(candle['open']),
        'high': parse_quotation(candle['high']),
        'low': parse_quotation(candle['low']),
        'close': parse_quotation(candle['close']),
        'volume': int(candle['volume']),
        'isComplete': candle['isComplete'],
        'candleSourceType': candle['candleSourceType']
    })

# Создаем DataFrame
df = pd.DataFrame(data)

# Преобразуем время в datetime
df['time'] = pd.to_datetime(df['time'])

# df['SMA_20'] = df['close'].rolling(window=20).mean()
# delta = df['close'].diff()
# gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
# loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# rs = gain / loss
# df['RSI'] = 100 - (100 / (1 + rs))

# high_low = df['high'] - df['low']
# high_close = (df['high'] - df['close'].shift()).abs()
# low_close = (df['low'] - df['close'].shift()).abs()
# true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
# df['ATR'] = true_range.rolling(window=14).mean()

# Устанавливаем время как индекс
df.set_index('time', inplace=True)

print(df)


from sklearn.preprocessing import StandardScaler

# Выбираем признаки для нормализации
features = ['open', 'high', 'low', 'close', 'volume']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


import numpy as np

# Преобразуем данные в numpy array
data = df[features].values

# Разделяем на обучающую и тестовую выборки
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Преобразуем данные для LSTM
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, 3])  # Предсказываем close
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, len(features))))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=10)

