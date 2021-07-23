import os
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model,layers,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,train_test_split
from tensorflow.keras.optimizers import SGD
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LeakyReLU
def sequential_model(end_train):
    model = tf.keras.Sequential()
    model.add(Dense(20, input_dim=end_train))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model