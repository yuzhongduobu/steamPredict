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
###############读取数据################
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
#shanchu=['V5','V14','V17','V21','V22','V25','V27','V32','V33','V34']
shanchu=[]
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
##################训练集、验证集划分#################
X, X_test, Y, Y_test = train_test_split(zhengqi_train.iloc[:,:38],zhengqi_train.iloc[:,38],test_size=0.2, random_state=1)
##################归一化###########################
X_min=X.min()
X_max=X.max()
X=(X-X_min)/(X_max-X_min)
Y_min=Y.min()
Y_max=Y.max()
Y=(Y-Y_min)/(Y_max-Y_min)

X_test_min=X_test.min()
X_test_max=X_test.max()
X_test=(X_test-X_test_min)/(X_test_max-X_test_min)
##################定义model#######################
_input1=Input(shape=[None,6])
Dense_1=Dense(6,activation='relu')(_input1)
Dropout_1=Dropout(0.2)(Dense_1)

_input2=Input(shape=[None,6])
Dense_2=Dense(6,activation='relu')(_input2)
Dropout_2=Dropout(0.2)(Dense_2)

_input3=Input(shape=[None,6])
Dense_3=Dense(6,activation='relu')(_input3)
Dropout_3=Dropout(0.2)(Dense_3)

_input4=Input(shape=[None,6])
Dense_4=Dense(6,activation='relu')(_input4)
Dropout_4=Dropout(0.2)(Dense_4)

_input5=Input(shape=[None,6])
Dense_5=Dense(6,activation='relu')(_input5)
Dropout_5=Dropout(0.2)(Dense_5)

_input6=Input(shape=[None,8])
Dense_6=Dense(8,activation='relu')(_input6)
Dropout_6=Dropout(0.2)(Dense_6)

concatenated = layers.concatenate([Dropout_1,Dropout_2,Dropout_3,Dropout_4,Dropout_5,Dropout_6],axis=-1)
Dropout_7=Dropout(0.2)(concatenated)
Dense_7=Dense(50,activation='relu')(Dropout_7)
Dropout_8=Dropout(0.2)(Dense_7)
Dense_8=Dense(50,activation='relu')(Dropout_8)
Dense_last=Dense(1)(Dense_8)
model = Model([_input1,_input2,_input3,_input4,_input5,_input6],Dense_last)
model.compile(loss='mean_squared_error', optimizer='adam')

history=model.fit([X.iloc[:,:6],X.iloc[:,6:12],X.iloc[:,12:18],X.iloc[:,18:24],X.iloc[:,24:30],X.iloc[:,30:38]], Y, epochs=500,validation_split=0.2,batch_size=32,verbose=1)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='upper left')
plt.show()


