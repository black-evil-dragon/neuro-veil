from re import S

from matplotlib import pyplot as plt
from data.fetcher import DataFetcher
from services.TinkoffService import TinkoffService
from services.InstrumentsService import InstrumentsService
from services.MarketDataService import MarketDataService

from utils.config import SANDBOX_TOKEN

from datetime import timedelta
from utils.time import now, prepare_date





import logging
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


logging.basicConfig(filename="./neuro.log", format='[%(asctime)s]  %(message)s', filemode='w')
logging.getLogger('neuro').addHandler(logging.StreamHandler(sys.stdout))

log = logging.getLogger('neuro')
log.setLevel(logging.DEBUG)


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


from utils.numbers import parse_quotation

class DataProcessor:
    @staticmethod
    def process_indicators(indicators):
        processed_data = []
        for indicator in indicators['technicalIndicators']:
            processed_data.append({
                'time': indicator['timestamp'],
                'value': parse_quotation(indicator['signal']),      
            })
        return processed_data
    
    
    @staticmethod
    def process_candles(candles):
        processed_data = []
        for candle in candles['candles']:
            processed_data.append({
                'time': candle['time'],
                'open': parse_quotation(candle['open']),
                'high': parse_quotation(candle['high']),
                'low': parse_quotation(candle['low']),
                'close': parse_quotation(candle['close']),
                'volume': int(candle['volume']),
                # 'isComplete': candle['isComplete'],
                # 'candleSourceType': candle['candleSourceType'],
                
            })
        return processed_data



def main():
    processor = DataProcessor()
    analyzer = DataAnalyzer()


    TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)

    instrumentsService = InstrumentsService(TService)
    marketDataService = MarketDataService(TService)

    response = instrumentsService.find_instrument('BBG0013HGFT4')
    [print(r) for r in response.get('instruments')]

    data = DataFetcher(TService=TService, processor=processor).get_data(
        instrument=response.get('instruments')[0],
        from_date=now() - timedelta(days=8000),
        to_date=now() - timedelta(days=7999),
        # additive_instruments=[
        #     moex
        # ]
    )

    [print(r) for r in data]





if __name__ == "__main__":
    main()