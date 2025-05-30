from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import tensorflow

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU, LayerNormalization, BatchNormalization, Attention, Concatenate, Bidirectional
from keras.layers import Input
from keras.mixed_precision import set_global_policy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import logging



from veil.neuro.model import NeuroModel
from veil.neuro.model.interface import InterfaceModel



log = logging.getLogger("neuro")
log.setLevel(logging.DEBUG)


class LstmModel(NeuroModel, InterfaceModel):

    def __str__(self):
        return 'LSTM model'

    def __init__(self):

        self.name = 'lstmModel'




    def build_model(self, input_shape: tuple) -> 'Sequential':
        return getattr(
            self,
            f"get_model_{self.version}"
        )(
            super().build_model(input_shape),
            input_shape
        )
    

    @staticmethod
    def get_model_default(model: 'Sequential', input_shape: tuple):
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape, activation="tanh"))
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))

        model.add(LSTM(100, return_sequences=False, activation="tanh"))
        # model.add(Dropout(0.2))
        
        model.add(Dense(150, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1, activation="linear"))

        optimizer = Adam(
            learning_rate=0.01
        )

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mae"]
        )

        return model
    
