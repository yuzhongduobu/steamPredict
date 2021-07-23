# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:05:55 2021

@author: Administrator
"""
import numpy
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#导入数据 
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
shanchu=['V27']
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
#去属性名，留值
train_value = zhengqi_train.values
#训练集、测试集输入、输出值提取
train_x = train_value[:,:end_train]
train_y = train_value[:,end_train]
drop_matrix=[0.15,0.15,0.15,0.15]
# define base mode
'''
def baseline_model():
    model = Sequential()
    model.add(Dense(end_train, input_dim=end_train))
    model.add(Dropout(drop_matrix[0]))
    model.add(Dense(65*(1-drop_matrix[0]), activation='relu'))
    model.add(Dropout(drop_matrix[1]))
    model.add(Dense(65*(1-drop_matrix[0])*(1-drop_matrix[1]), activation='relu'))
    model.add(Dropout(drop_matrix[2]))
    model.add(Dense(65*(1-drop_matrix[0])*(1-drop_matrix[1])*(1-drop_matrix[2]), activation='relu'))
    model.add(Dropout(drop_matrix[3]))
    model.add(Dense(65*(1-drop_matrix[0])*(1-drop_matrix[1])*(1-drop_matrix[2])*(1-drop_matrix[3]), activation='tanh', use_bias=True))
    model.add(Dense(1, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
'''
def baseline_model():
    model = Sequential()
    model.add(Dense(end_train, input_dim=end_train))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(45, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(40, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(36, activation='tanh', use_bias=True))
    model.add(Dense(1, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

# fix random seed for reproducibility
seed = end_train
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=500,batch_size=25, verbose=1)))
pipeline = Pipeline(estimators)   
#10折交叉验证
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, train_x, train_y, cv=kfold)
print("Results: 平均值%.2f (标准差%.2f) MSE" % (results.mean(), results.std()))