import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class DataAnalyzer:
    @staticmethod
    def create_dataframe(data):
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

    @staticmethod
    def normalize_data(df, features):
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def prepare_lstm_data(data, look_back=60):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back, 3])
        return np.array(X), np.array(y)

    @staticmethod
    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model