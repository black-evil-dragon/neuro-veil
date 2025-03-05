from deprecated import deprecated

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

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
import matplotlib.pyplot as plt



log = logging.getLogger('neuro')
log.setLevel(logging.DEBUG)

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



class Model:

    model = None 

    mean = None
    std = None


    def __init__(self):
        self.processor = DataProcessor()
        self.scaler = StandardScaler()
        self.scaler_minmax = MinMaxScaler()


        self.SHOW_CORRELATION = False
        self.SHOW_METRICS = False
        self.SHOW_GRAPHICS = True

        self.look_back = LOOK_BACK
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS


        self.train_size = 0.8
        self.val_size = 0.1


        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]

        self.target = 'close'


        log.info("Инициализирован Model")



    def train(self, data, features):
        log.info("Начало обучения модели")
        df = self.create_dataframe(data)

        if df.isnull().values.any():
            log.warning("Обнаружены пропущенные значения (NaN). Заполняем нулями.")
            df.fillna(0, inplace=True)


        data = df[features].values
        target = df[self.target].values

        normalized_data, normalized_target = self.normalize_data(data, target)


        # Разделение данных
        X, y = self.create_sequences(normalized_data, normalized_target, self.look_back)

        train_size = int(len(X) * self.train_size)
        val_size = int(len(X) * self.val_size)
    
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

        log.debug(f"Данные нормализованы. Размер данных: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")



        # Кореляция признаков с целевой переменной
        if self.SHOW_CORRELATION:
            self.show_correlation(df, features)


        log.info("Модель построена. Начало обучения...")
        # Построение и обучение модели
        model = self.build_lstm_model((self.look_back, len(features)))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),

            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks
        )
        self.model = model
        log.info("Обучение завершено")



        y_pred = model.predict(X_test)

        # Оценка на тестовых данных
        if self.SHOW_METRICS:
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)


            log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")

            try:
                plt.figure(figsize=(10, 6))

                # График тренировочной ошибки
                plt.plot(history.history['loss'], label='Тренировочная ошибка (loss)', color='blue')

                # График валидационной ошибки
                plt.plot(history.history['val_loss'], label='Валидационная ошибка (val_loss)', color='red')

                # Настройка графика
                plt.title('График ошибок в зависимости от эпохи')
                plt.xlabel('Эпохи')
                plt.ylabel('Ошибка (Loss)')
                plt.legend()
                plt.grid(True)
                plt.show()
            except:
                pass


        # Визуализация результатов
        if self.SHOW_GRAPHICS:
            y_test_original = self.scaler_minmax.inverse_transform(y_test.reshape(-1, 1))
            y_pred_original = self.scaler_minmax.inverse_transform(y_pred)

            plt.figure(figsize=(14, 5))
            plt.plot(y_test_original, color='blue', label='Фактические значения')
            plt.plot(y_pred_original, color='red', label='Предсказанные значения')
            plt.title('Предсказание vs Фактические значения')
            plt.xlabel('Время')
            plt.ylabel('Цена закрытия')
            plt.legend()
            plt.show()



    def predict(self, data, features):
        """
        Выполняет предсказание на основе данных.
        
        :param data: Данные для предсказания
        :param features: Список признаков, используемых для предсказания.
        :return: model.
        """
        if self.model is None:
            log.warning("Модель не загружена. Используйте метод load() для загрузки модели.")
            return None

        log.info("Начало предсказания")
        df = self.create_dataframe(data)
        df = self.normalize_data(df, features)
        prepared_data = df[features].values

        look_back = self.look_back  # Должно совпадать с look_back при обучении
        X, _ = self.create_sequences(prepared_data, look_back)

        log.debug(f"Данные подготовлены для предсказания. Форма X: {X.shape}")
        y_pred = self.model.predict(X)

        log.info("Предсказание завершено")
        return y_pred
    




    def normalize_data(self, data, target):
        scaled_data = self.scaler_minmax.fit_transform(data)
        scaled_target = self.scaler_minmax.fit_transform(target.reshape(-1, 1))

        return scaled_data, scaled_target
    


    @deprecated()
    def denormalize(self, predicted):
        return predicted * self.std + self.mean


    @staticmethod
    def create_dataframe(data):
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

    @staticmethod
    def create_sequences(data, target, sequence_length=60):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    


    def build_lstm_model(self, input_shape):
        set_global_policy('mixed_float16')

        model = Sequential()
        
        model = self.get_model_v2(model, input_shape)
        self.model = model

        return model
    

    @staticmethod
    def get_model_v1(model, input_shape):
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        return model
    
    @staticmethod
    def get_model_v2(model, input_shape):
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    

    @staticmethod
    def show_correlation(df, features):
        # Постройте heatmap корреляции
        corr = df[features].corr()

        print(corr)

        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()


    def load(self, model=None, name: str=None, extension: str=None, test: bool=False) -> bool:

        if model is not None:
            log.info(f"Загрузка модели: {model}")

            self.model = model

        elif name is not None and extension is not None:
            path = f'./output/models/{name}/{"test" if test else ""}/model.{extension}'

            log.info(f"Загрузка модели: {path}")

            self.model = tf.keras.models.load_model(path)

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

        
