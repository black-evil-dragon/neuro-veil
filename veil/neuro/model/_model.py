# # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # from sklearn.model_selection import TimeSeriesSplit

# # import tensorflow as tf
# # from keras.models import Sequential


# # from keras.layers import LSTM
# # from keras.layers import Dense
# # from keras.layers import Dropout
# # from keras.layers import LeakyReLU, LayerNormalization, BatchNormalization, Attention, Concatenate, Bidirectional
# # from keras.layers import Input


# # from keras.mixed_precision import set_global_policy
# # from keras.optimizers import Adam
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


# # from data.common import InstrumentDataModel
# # from utils.config import BATCH_SIZE, EPOCHS, LOOK_BACK


# # import numpy as np
# # import pandas as pd
# # import logging
# # import os
# # import matplotlib.pyplot as plt
# # import pickle
# # import seaborn as sns


# # log = logging.getLogger("neuro")
# # log.setLevel(logging.DEBUG)

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# class Model:
#     model = None

#     mean = None
#     std = None

#     def __init__(self):
#         set_global_policy(
#             'mixed_float16'
#             # "float32"
#         )

#         self.scaler = StandardScaler()
#         self.scaler_minmax = MinMaxScaler()

#         self.SHOW_CORRELATION = False
#         self.SHOW_METRICS = False
#         self.SHOW_GRAPHICS = True

#         self.look_back = LOOK_BACK
#         self.batch_size = BATCH_SIZE
#         self.epochs = EPOCHS

#         self.lstm_version = 3

#         self.train_size = 0.8
#         self.val_size = 0.1

#         self.callbacks = [
#             EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
#             ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
#         ]

#         self.target = "close"

#         log.debug("Инициализирован Model")




#     def set_look_back(self, value):
#         self.look_back = value
#         log.info(f"Установлено значение лага: {value}")

#     def set_batch_size(self, value):
#         self.batch_size = value
#         log.info(f"Установлено значение размера батча: {value}")

#     def set_epochs(self, value):
#         self.epochs = value
#         log.info(f"Установлено значение эпох: {value}")

#     def set_model_version(self, value):
#         self.lstm_version = value
#         log.info(f"Установлено значение модели: {value}")



#     def reset(self):
#         self.__init__()



#     #########################################################################################################
#     #
#     # Метод для обучения модели
#     #
#     ##########################################################################################################
#     def train(self, train_data, features):
#         log.info("Начало обучения модели")


#         df = InstrumentDataModel.dict_to_dataframe(data=train_data)

#         columns = ['close', 'IMOEXF', 'USD000UTSTOM']
        
#         for column in columns:
#             df = InstrumentDataModel.differentiate_data(df, column=column,)
#             features += [f'{column}_diff_1']


#         data = df[features].values
#         target = df[self.target].values
#         dates = pd.to_datetime(df.index)


#         normalized_data, normalized_target = self.normalize_data(data, target)

#         # Разделение данных
#         X, y = self.create_sequences(normalized_data, normalized_target, self.look_back)

#         train_size = int(len(X) * self.train_size)
#         # val_size = int(len(X) * self.val_size)

#         X_train, X_test = (
#             X[:train_size],
#             X[train_size:],
#         )
#         y_train, y_test = (
#             y[:train_size],
#             y[train_size:],
#         )

#         log.debug(
#             f"Данные нормализованы. Размер данных: train={len(X_train)}, test={len(X_test)}"
#         )


#         # tscv = TimeSeriesSplit(n_splits=5)
#         # for train_index, val_index in tscv.split(X):
#         #     X_train, X_val = X[train_index], X[val_index]
#         #     y_train, y_val = y[train_index], y[val_index]
#         # log.debug(
#         #     "Проведена кроссвалидация"
#         # )


#         # Кореляция признаков с целевой переменной
#         if self.SHOW_CORRELATION:
#             self.show_correlation(df, features)



#         log.info("Модель построена. Начало обучения...")
#         model = self.build_lstm_model((self.look_back, len(features)))
#         history = model.fit(
#             X_train, y_train,
#             # validation_data=(X_val, y_val),
#             validation_split=0.2,

#             batch_size=self.batch_size,
#             epochs=self.epochs,

#             callbacks=self.callbacks,
#         )
#         self.model = model
#         log.info("Обучение завершено")

#         y_pred = model.predict(X_test)

#         # Оценка на тестовых данных
#         if self.SHOW_METRICS:
#             mse = mean_squared_error(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)

#             log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")

#             try:
#                 plt.figure(figsize=(10, 6))

#                 # График тренировочной ошибки
#                 plt.plot(
#                     history.history["loss"],
#                     label="Тренировочная ошибка (loss)",
#                     color="blue",
#                 )

#                 # График валидационной ошибки
#                 plt.plot(
#                     history.history["val_loss"],
#                     label="Валидационная ошибка (val_loss)",
#                     color="red",
#                 )

#                 plt.title("График ошибок в зависимости от эпохи")
#                 plt.xlabel("Эпохи")
#                 plt.ylabel("Ошибка (Loss)")
#                 plt.legend()
#                 plt.grid(True)
#                 plt.show()
#             except:
#                 log.info(f"Метрики модели на тестовых данных: MSE={mse:.4f}, MAE={mae:.4f}")

#         # Визуализация результатов
#         if self.SHOW_GRAPHICS:
#             y_test_original = self.scaler_minmax.inverse_transform(
#                 y_test.reshape(-1, 1)
#             )
#             y_pred_original = self.scaler_minmax.inverse_transform(y_pred)

#             plt.figure(figsize=(14, 5))
#             plt.plot(
#                 dates[-len(y_test) :],
#                 y_test_original,
#                 color="blue",
#                 label="Фактические значения",
#             )
#             plt.plot(
#                 dates[-len(y_test) :],
#                 y_pred_original,
#                 color="red",
#                 label="Предсказанные значения",
#             )
#             plt.title(f'Look Back: {self.look_back}, Batch Size: {self.batch_size}, Epochs: {self.epochs}, LSTM Version: {self.lstm_version}')
#             plt.xlabel("Время")
#             plt.ylabel("Цена закрытия")
#             plt.gcf().autofmt_xdate()
#             plt.legend()
#             # plt.show()

#             plt.savefig(f'./tests/lb{self.look_back}_bs{self.batch_size}_ep{self.epochs}_v{self.lstm_version}.png')
#             plt.close()




#     #########################################################################################################
#     #
#     # Метод для
#     #
#     ##########################################################################################################
#     def predict_future(self, initial_input, real_data, steps):
#         predictions = []
#         current_input = initial_input.copy()

#         for i in range(steps):

#             next_prediction = self.model.predict(
#                 current_input.reshape(1, self.look_back, -1)
#             )


#             predictions.append(next_prediction[0][0])


#             current_input = np.roll(current_input, -1, axis=0)
#             if i < len(real_data):
#                 current_input[-1] = real_data[i]
#             else:
#                 current_input[-1] = next_prediction[0]

#         return np.array(predictions)


#     def predict(self, data, features, additive_steps=0):
#         if self.model is None:
#             log.warning(
#                 "Модель не загружена. Используйте метод load() для загрузки модели."
#             )
#             return None

#         # # Создание DataFrame
#         # df = self.create_dataframe(data)

#         # half_size = int(len(df) * 0.5)


#         # # Проверка на пропущенные значения
#         # if df.isnull().values.any():
#         #     log.warning("Обнаружены пропущенные значения (NaN). Заполняем нулями.")
#         #     df.fillna(0, inplace=True)

#         # print(df)

#         # # Разделение данных на две половины
#         # first_half = df.iloc[:half_size]
#         # second_half = df.iloc[half_size:]

#         # # ====================================
#         # # Подготовка данных для первой половины
#         # # ====================================
#         # first_half_data = first_half[features].values
#         # first_half_target = first_half["close"].values
#         # first_half_dates = first_half.index

#         # # Нормализация данных первой половины
#         # normalized_first_half_data = self.scaler_minmax.fit_transform(first_half_data)
#         # normalized_first_half_target = self.scaler_minmax.fit_transform(
#         #     first_half_target.reshape(-1, 1)
#         # )

#         # # Создание последовательностей для первой половины
#         # X_first, y_first = self.create_sequences(
#         #     normalized_first_half_data, normalized_first_half_target, self.look_back
#         # )

#         # # Прогнозирование на первой половине
#         # first_half_predictions = self.model.predict(X_first)
#         # first_half_predictions = self.scaler_minmax.inverse_transform(
#         #     first_half_predictions
#         # )
#         # y_first = self.scaler_minmax.inverse_transform(y_first)
#         # log.info("Прогнозирование на первой половине данных завершено")


#         # # ===============================================
#         # # Подготовка данных для второй половины (авторегрессия)
#         # # ===============================================
#         # second_half_data = second_half[features].values
#         # second_half_target = second_half["close"].values
#         # second_half_dates = second_half.index

#         # # Нормализация данных второй половины
#         # normalized_second_half_data = self.scaler_minmax.fit_transform(second_half_data)

#         # # Начальное окно для авторегрессии (последние look_back свечей из первой половины)
#         # initial_window = normalized_first_half_data[-self.look_back :]

#         # # Рекурсивное прогнозирование для второй половины с использованием реальных данных
#         # second_half_predictions_normalized = self.predict_future(
#         #     initial_window,
#         #     normalized_second_half_data,
#         #     len(second_half_data) + additive_steps
#         # )
#         # dummy_features = np.zeros(
#         #     (len(second_half_predictions_normalized), len(features) - 1)
#         # )  # 4 других признака
#         # predictions_with_dummy_features = np.hstack(
#         #     (second_half_predictions_normalized.reshape(-1, 1), dummy_features)
#         # )

#         # # Применяем inverse_transform
#         # second_half_predictions = self.scaler_minmax.inverse_transform(
#         #     predictions_with_dummy_features
#         # )

#         # # Извлекаем только предсказанные значения для close
#         # second_half_predictions = second_half_predictions[:, 0]
#         # log.info("Прогнозирование на второй половине данных завершено")



#         # # Генерация дополнительных дат
#         # additional_dates = pd.date_range(
#         #     start=second_half_dates[-1] + pd.Timedelta(hours=1),  # Следующий час после последней даты
#         #     periods=additive_steps,  # Количество дополнительных шагов
#         #     freq="h"  # Частота (например, "H" для часов)
#         # )

#         # # Объединяем исходные даты с дополнительными
#         # extended_second_half_dates = np.concatenate([second_half_dates, additional_dates])

#         # extended_second_half_true = np.pad(
#         #     second_half_target,
#         #     (0, additive_steps),  # Добавляем 10 значений
#         #     mode="constant",
#         #     constant_values=np.nan
#         # )

#         # # Визуализация результатов
#         # self.plot_predictions(
#         #     first_half_dates[self.look_back :],
#         #     y_first,
#         #     first_half_predictions,
#         #     extended_second_half_dates,
#         #     # second_half_target,
#         #     extended_second_half_true,
#         #     second_half_predictions,
#         # )




#     #########################################################################################################
#     #
#     # Хелперы для формирования данных, нормализации и обучения модели
#     #
#     ##########################################################################################################
#     def normalize_data(self, data, target):
#         scaled_data = self.scaler_minmax.fit_transform(data)
#         scaled_target = self.scaler_minmax.fit_transform(target.reshape(-1, 1))

#         return scaled_data, scaled_target

#     @staticmethod
#     def create_dataframe(data):
#         df = pd.DataFrame(data)
#         df["time"] = pd.to_datetime(df["time"])
#         df.set_index("time", inplace=True)
#         return df

#     @staticmethod
#     def create_sequences(data, target, sequence_length=60):
#         X, y = [], []
#         for i in range(len(data) - sequence_length):
#             X.append(data[i : i + sequence_length])
#             y.append(target[i + sequence_length])
#         return np.array(X), np.array(y)

#     @staticmethod
#     def show_correlation(df, features):
#         # Постройте heatmap корреляции
#         corr = df[features].corr()

#         print(corr)

#         sns.heatmap(corr, annot=True, cmap="coolwarm")
#         plt.show()

#     @staticmethod
#     def plot_predictions(
#         first_half_dates,
#         first_half_true,
#         first_half_preds,
#         second_half_dates,
#         second_half_true,
#         second_half_preds,
#     ):
#         plt.figure(figsize=(14, 5))

#         # Реальные значения (первая половина)
#         plt.plot(
#             first_half_dates,
#             first_half_true,
#             label="Реальные значения (первая половина)",
#             color="blue",
#         )

#         # Предсказанные значения (первая половина)
#         plt.plot(
#             first_half_dates,
#             first_half_preds,
#             label="Предсказанные значения (первая половина)",
#             color="orange",
#             linestyle="-",
#         )

#         # Реальные значения (вторая половина)
#         plt.plot(
#             second_half_dates,
#             second_half_true,
#             label="Реальные значения (вторая половина)",
#             color="green",
#             linestyle="--",
#         )

#         # Предсказанные значения (вторая половина, авторегрессия)
#         plt.plot(
#             second_half_dates,
#             second_half_preds,
#             label="Предсказанные значения (вторая половина, c авторегрессией)",
#             color="red",
#             linestyle="-",
#         )

#         plt.xlabel("Время")
#         plt.ylabel("Цена закрытия")
#         plt.title("Прогнозирование цен на акции")
#         plt.legend()
#         plt.gcf().autofmt_xdate()
#         plt.show()



#     #########################################################################################################
#     #
#     # Формирование структуры модели
#     #
#     ##########################################################################################################
#     def build_lstm_model(self, input_shape):
#         model = Sequential()


#         getattr(self, f"get_model_v{self.lstm_version}")(model, input_shape)

#         self.model = model

#         return model

#     @staticmethod
#     def get_model_v1(model, input_shape):
#         """
#         Слишком прост, неэффективен
#         """
#         model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
#         model.add(LSTM(50, return_sequences=False))
#         model.add(Dense(25))
#         model.add(Dense(1))

#         model.compile(optimizer="adam", loss="mean_squared_error")

#         return model
    

#     @staticmethod
#     def get_model_v2(model, input_shape):
#         """
#         Показал неточные результаты
#         """
#         model.add(LSTM(100, return_sequences=True, input_shape=input_shape, activation="tanh"))
#         model.add(Dropout(0.2))

#         model.add(LSTM(100, return_sequences=False, activation="tanh"))
#         model.add(Dropout(0.2))

#         model.add(Dense(50, activation="relu"))
#         model.add(Dense(1, activation="linear"))

#         optimizer = Adam(
#             learning_rate=0.05
#         )

#         model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

#         return model

#     @staticmethod
#     def get_model_v3(model, input_shape):
#         """
#         Хорошо справился с задачей, добавление нового слоя привело к более точному анализу
#         """
#         model.add(LSTM(100, return_sequences=True, input_shape=input_shape, activation="tanh"))
#         model.add(LayerNormalization())
#         model.add(LeakyReLU(alpha=0.1))
#         model.add(Dropout(0.2))

#         model.add(LSTM(100, return_sequences=False, activation="tanh"))
#         # model.add(BatchNormalization())
#         # model.add(LeakyReLU(alpha=0.1))
#         model.add(Dropout(0.2))
        
#         model.add(Dense(150, activation="relu"))
#         # model.add(LeakyReLU(alpha=0.1))
#         model.add(Dense(50, activation="relu"))
#         model.add(Dense(1, activation="linear"))

#         optimizer = Adam(
#             learning_rate=0.01
#         )

#         model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

#         return model
    

#     @staticmethod
#     def get_model_v4(model, input_shape):
#         """
#         Экспериментальный
#         """
#         # model.add(Input(shape=input_shape, dtype='float32'))

#         model.add(
#             Bidirectional(
#                 LSTM(64, return_sequences=True), 
#                 input_shape=input_shape
#             )
#         )
#         model.add(LayerNormalization())
#         model.add(LeakyReLU(alpha=0.2))

#         model.add(LSTM(128, return_sequences=False))
#         model.add(LayerNormalization())
#         model.add(Dropout(0.3))

        
#         model.add(Dense(128, activation='relu'))
#         model.add(Dropout(0.3))

#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='linear'))


#         optimizer = Adam(
#             learning_rate=0.005,
#             decay=1e-4
#         )
        
#         # Изменение 7: Добавление Huber loss для устойчивости к выбросам
#         model.compile(
#             optimizer=optimizer, 
#             loss=tf.keras.losses.Huber(delta=1.5),  # Комбинация MAE и MSE
#             metrics=["mae", "mse"]
#         )
        
#         return model




#     #########################################################################################################
#     #
#     # Хелперы для сохранения и загрузки модели
#     #
#     ##########################################################################################################
#     def load(self, model=None, name: str = None, extension: str = None, test: bool = False) -> bool:
#         if model is not None:
#             log.info(f"Загрузка модели: {model}")
#             self.model = model
#         elif name is not None and extension is not None:
#             path = f"./output/models/{name}/{'test' if test else ''}/model.{extension}"
#             log.info(f"Загрузка модели: {path}")
#             self.model = tf.keras.models.load_model(path)

#             # Загрузка scaler_minmax
#             scaler_path = f"./output/models/{name}/{'test' if test else ''}/scaler_minmax.pkl"
#             if os.path.exists(scaler_path):
#                 with open(scaler_path, "rb") as f:
#                     self.scaler_minmax = pickle.load(f)
#                 log.info("scaler_minmax успешно загружен")
#             else:
#                 log.warning("Файл scaler_minmax.pkl не найден. scaler_minmax не загружен.")
#         else:
#             log.warning("Неверные параметры для загрузки модели. Используйте или модель, или имя и расширение.")
#             return False

#         log.info("Модель успешно загружена")
#         return True

#     def save(self, name: str, model=None, test=False):
#         path = f"./output/models/{name}/{'test' if test else ''}/"
#         os.makedirs(path, exist_ok=True)  # Создаем директорию, если она не существует

#         if model:
#             log.info(f"Сохранение модели в файл: {path}model.keras(.h5)")
#             model.save(f"{path}model.keras")
#             model.save(f"{path}model.h5")
#         elif self.model:
#             log.info(f"Сохранение модели в файл: {path}model.keras(.h5)")
#             self.model.save(f"{path}model.keras")
#             self.model.save(f"{path}model.h5")
#         else:
#             log.warning("Модель не загружена. Используйте метод load() для загрузки модели.")
#             return

#         # Сохранение scaler_minmax
#         if hasattr(self, 'scaler_minmax'):
#             with open(f"{path}scaler_minmax.pkl", "wb") as f:
#                 pickle.dump(self.scaler_minmax, f)
#             log.info("scaler_minmax успешно сохранен")
#         else:
#             log.warning("scaler_minmax не найден. Сохранение scaler_minmax пропущено.")

#         log.info("Модель и scaler_minmax успешно сохранены")
