from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data.proccessor import DataProcessor

from utils.config import BATCH_SIZE, EPOCHS, LOOK_BACK

import numpy as np
import pandas as pd
import logging
import os



log = logging.getLogger('neuro')
log.setLevel(logging.DEBUG)

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



class Model:

    model = None 


    def __init__(self):
        self.processor = DataProcessor()
        self.scaler = StandardScaler()
    
        self.mean = None
        self.std = None

        self.look_back = LOOK_BACK
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS

        log.info("Инициализирован Model")



    def train(self, data, features):
        log.info("Начало обучения модели")
        df = self.create_dataframe(data)
        df = self.normalize_data(df, features)
        log.debug(f"Данные нормализованы. Размер данных: {df.shape}")

        prepared_data = df[features].values
        train_size = int(len(prepared_data) * 0.7)
        val_size = int(len(prepared_data) * 0.15)

        train = prepared_data[:train_size]
        val = prepared_data[train_size:train_size + val_size]
        test = prepared_data[train_size + val_size:]

        log.debug(f"Данные разделены на train/val/test: {len(train)}/{len(val)}/{len(test)}")

        X_train, y_train = self.prepare_lstm_data(train, self.look_back)
        X_val, y_val = self.prepare_lstm_data(val, self.look_back)
        X_test, y_test = self.prepare_lstm_data(test, self.look_back)

        log.debug(f"Данные подготовлены для LSTM. Форма X_train: {X_train.shape}, X_val: {X_val.shape}")

        model = self.build_lstm_model((self.look_back, len(features)))

        # Callback'и
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        log.info("Модель построена. Начало обучения...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping, reduce_lr]
        )

        log.info("Обучение завершено")

        # Оценка на тестовых данных
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")

        self.model = model



    def predict(self, data, features):
        """
        Выполняет предсказание на основе данных.
        
        :param data: Данные для предсказания (список словарей или DataFrame).
        :param features: Список признаков, используемых для предсказания.
        :return: Предсказанные значения.
        """
        if self.model is None:
            log.warning("Модель не загружена. Используйте метод load() для загрузки модели.")
            return None

        log.info("Начало предсказания")
        df = self.create_dataframe(data)
        df = self.normalize_data(df, features)
        prepared_data = df[features].values

        look_back = 120  # Должно совпадать с look_back при обучении
        X, _ = self.prepare_lstm_data(prepared_data, look_back)

        log.debug(f"Данные подготовлены для предсказания. Форма X: {X.shape}")
        y_pred = self.model.predict(X)

        log.info("Предсказание завершено")
        return y_pred
    


    def load(self, model=None, name: str=None, extension: str=None) -> bool:
        if model is not None:
            log.info(f"Загрузка модели: {model}")

            self.model = model

        elif name is not None and extension is not None:
            log.info(f"Загрузка модели: {name}.{extension}")

            self.model = tf.keras.models.load_model(f'./output/{name}.{extension}')

        else:
            log.warning("Неверные параметры для загрузки модели. Используйте или модель, или имя и расширение.")
            return False
        
        log.info("Модель успешно загружена")
        return True


    def save(self, name:str, model=None, test=False):
        path = '' if not test else '/test/'
        if model:
            log.info(f"Сохранение модели в файл: ./output/models/{name}{path}/model.keras(.h5)")

            model.save(f'./output/models/{name}{path}/model.keras')
            model.save(f'./output/models/{name}{path}/model.h5')

            log.info("Модель успешно сохранена")
        elif self.model:
            log.info(f"Сохранение модели в файл: ./output/models/{name}{path}/model.keras(.h5)")

            self.model.save(f'./output/models/{name}{path}/model.keras')
            self.model.save(f'./output/models/{name}{path}/model.h5')

            log.info("Модель успешно сохранена")
        else:
            log.warning("Модель не загружена. Используйте метод load() для загрузки модели.")


    def normalize_data(self, df, features):
        """
        Нормализация данных с использованием StandardScaler.
        :param df: DataFrame с данными.
        :param features: Список фичей для нормализации.
        :return: Нормализованный DataFrame.
        """
        df[features] = self.scaler.fit_transform(df[features])
        self.mean = self.scaler.mean_[features.index('close')]  # Сохраняем среднее для 'close'
        self.std = self.scaler.scale_[features.index('close')]  # Сохраняем стандартное отклонение для 'close'
        df.fillna(0, inplace=True)
        return df
    
    def denormalize(self, predicted):
        """
        Денормализация предсказанных значений.
        :param predicted: Предсказанные значения (в нормализованном виде).
        :return: Денормализованные значения.
        """
        return predicted * self.std + self.mean


    @staticmethod
    def create_dataframe(data):
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

    @staticmethod
    def prepare_lstm_data(data, look_back=60):
        X, y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back + 1, 3])
        return np.array(X), np.array(y)
    


    @staticmethod
    def build_lstm_model(input_shape):
        set_global_policy('mixed_float16')

        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model
        
