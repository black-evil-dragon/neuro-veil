from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU, LayerNormalization, BatchNormalization, Attention, Concatenate, Bidirectional
from keras.layers import Input


from keras.mixed_precision import set_global_policy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


from veil.neuro.model.interface import InterfaceModel
from veil.tinkoff.data.common import InstrumentDataModel



log = logging.getLogger("neuro")
log.setLevel(logging.DEBUG)



class NeuroModel:
    name = 'neuroModel'
    target = "close"
    version = "default"

    SHOW_CORRELATION = False
    SHOW_METRICS = False
    SHOW_GRAPHICS = True
    SAVE_RESULT = False

    # Model settings
    prototype: 'InterfaceModel' = None
    train_size = 0.8
    val_split = 0.2
    look_back = 60
    batch_size = 64
    epochs = 10


    # Scaler
    scaler = MinMaxScaler()


    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
    ]


    def __str__(self):
        return ""


    def __init__(self, prototype):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Set config params
        try:
            from veil.utils.config import BATCH_SIZE, EPOCHS, LOOK_BACK

            self.set_batch_size(BATCH_SIZE)
            self.set_epochs(EPOCHS)
            self.set_look_back(LOOK_BACK)

        except Exception:
            log.exception("Не удалось получить параметры обучения модели из файла config.py")


        # Initialize model
        try:
            self.set_model(prototype=prototype())
        except Exception:
            log.exception('Не удалось инициализировать модель')
            exit()



    #* _________________________________________________________________________
    #* | Геттеры и сеттеры                                                      |
    def set_policy(self, policy):
        set_global_policy(policy)
        log.info(f"Установлено значение: {policy}")

    def set_scaler(self, scaler):
        self.scaler = scaler
        log.info(f"Установлено значение: {scaler}")

    def set_look_back(self, value: int):
        self.look_back = value
        log.info(f"Установлено значение look_back: {value}")

    def set_batch_size(self, value: int):
        self.batch_size = value
        log.info(f"Установлено значение размера батча: {value}")

    def set_epochs(self, value: int):
        self.epochs = value
        log.info(f"Установлено значение эпох: {value}")

    def set_model(self, prototype: 'InterfaceModel'):
        self.prototype = prototype
        log.info(f"Установлена модель нейросети: {prototype.name}")

    def set_features(self, features: list):
        self.features = features
        log.info(f"Установлено значение features: {features}")

    #* |________________________________________________________________________|


    #* __________________________________________________________________________
    #* | Обучение                                                                |
    def train(self, train_data, features):
        log.info("Начало обучения модели")
        df = InstrumentDataModel.dict_to_dataframe(data=train_data)


        # Кореляция признаков с целевой переменной
        if self.SHOW_CORRELATION:
            self.show_correlation(df, features)


        # Разделение данных на признаки и целевую переменную
        data = df[features].values
        target = df[self.target].values
        dates = pd.to_datetime(df.index)

        # Нормализация данных
        normalized_data, normalized_target = self.normalize_data(data, target)


        # Разделение данных
        X, y = self.create_sequences(normalized_data, normalized_target, self.look_back)


        # Разделение на тренировочную и тестовую выборки
        train_size = int(len(X) * self.train_size)
        X_train, X_test = (
            X[:train_size],
            X[train_size:],
        )
        y_train, y_test = (
            y[:train_size],
            y[train_size:],
        )

        log.debug(f"Данные нормализованы. Размер данных: train={len(X_train)}, test={len(X_test)}")

 
        model = self.prototype.build_model((self.look_back, len(features)))
        log.info("Модель построена. Начало обучения...")



        history = model.fit(
            X_train, y_train,

            validation_split=self.val_split,

            batch_size=self.batch_size,
            epochs=self.epochs,

            callbacks=self.callbacks,
        )
        log.info("Обучение завершено")


        y_pred = model.predict(X_test)

        # Оценка на тестовых данных
        if self.SHOW_METRICS:
            mse, mae = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)

            log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")

            try:
                plt.figure(figsize=(10, 6))

                # График тренировочной ошибки
                plt.plot(
                    history.history["loss"],
                    label="Тренировочная ошибка (loss)",
                    color="blue",
                )

                # График валидационной ошибки
                plt.plot(
                    history.history["val_loss"],
                    label="Валидационная ошибка (val_loss)",
                    color="red",
                )

                plt.title("График ошибок в зависимости от эпохи")
                plt.xlabel("Эпохи")
                plt.ylabel("Ошибка (Loss)")
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception:
                log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")


        # Визуализация результатов
        if self.SHOW_GRAPHICS:
            y_test_original = self.scaler.inverse_transform(
                y_test.reshape(-1, 1)
            )
            y_pred_original = self.scaler.inverse_transform(y_pred)

            plt.figure(figsize=(14, 5))
            plt.plot(
                dates[-len(y_test) :],
                y_test_original,
                color="blue",
                label="Фактические значения",
            )
            plt.plot(
                dates[-len(y_test) :],
                y_pred_original,
                color="red",
                label="Предсказанные значения",
            )
            plt.title(f'Look Back: {self.look_back}, Batch Size: {self.batch_size}, Epochs: {self.epochs}, Pack version: {self.version}')
            plt.xlabel("Время")
            plt.ylabel("Цена закрытия")
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.show()

            # plt.savefig(f'./tests/lb{self.look_back}_bs{self.batch_size}_ep{self.epochs}_{self.version}.png')
            plt.close()


    #* __________________________________________________________________________
    #* | Базовый метод                                                           |
    #? | Нужен для                                                          |
    def build_model(self, input_shape):
        return Sequential()



    #* __________________________________________________________________________
    #* | Хелперы                                                                 |
    def normalize_data(self, data, target):
        scaled_data = self.scaler.fit_transform(data)
        scaled_target = self.scaler.fit_transform(target.reshape(-1, 1))

        return scaled_data, scaled_target
    

    @staticmethod
    def create_sequences(data, target, sequence_length=60):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)

    @staticmethod
    def show_correlation(df, features):
        corr = df[features].corr()

        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.show()


    def load(self, model=None, name: str = None, extension: str = None, test: bool = False) -> bool:
        if model is not None:
            log.info(f"Загрузка модели: {model}")
            self.model = model
        elif name is not None and extension is not None:
            path = f"./output/models/{name}/{'test' if test else ''}/model.{extension}"
            log.info(f"Загрузка модели: {path}")
            self.model = tf.keras.models.load_model(path)

            # Загрузка scaler_minmax
            scaler_path = f"./output/models/{name}/{'test' if test else ''}/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler_minmax = pickle.load(f)
                log.info("scaler_minmax успешно загружен")
            else:
                log.warning("Файл scaler.pkl не найден. scaler не загружен.")
        else:
            log.warning("Неверные параметры для загрузки модели. Используйте или модель, или имя и расширение.")
            return False

        log.info("Модель успешно загружена")
        return True


    def save(self, name: str, model=None, test=False):
        path = f"./output/models/{name}/{'test' if test else ''}/"
        os.makedirs(path, exist_ok=True)

        if model:
            log.info(f"Сохранение модели в файл: {path}model.keras(.h5)")
            model.save(f"{path}model.keras")
            model.save(f"{path}model.h5")
        elif self.model:
            log.info(f"Сохранение модели в файл: {path}model.keras(.h5)")
            self.model.save(f"{path}model.keras")
            self.model.save(f"{path}model.h5")
        else:
            log.warning("Модель не загружена. Используйте метод load() для загрузки модели.")
            return


        if hasattr(self, 'scaler'):
            with open(f"{path}scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            log.info("scaler_minmax успешно сохранен")
        else:
            log.warning("scaler не найден. Сохранение scaler пропущено.")

        log.info("Модель и scaler успешно сохранены")
    #* |
    #* |_________________________________________________________________________|