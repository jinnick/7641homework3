# Import General Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import metrics
# Loading Data
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
def cnn_train(predictors,target):
  
    n_cols = predictors.shape[1]
    print('training start')
    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape = (n_cols,)))
    #model.add(Dense(25, activation='relu'))
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Dropout(0.5))
    #model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
    history = model.fit(predictors, target, epochs=50, validation_split=0.5)
    return model,history