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
from sequential import sequential_model
from read_txt_PCA import read_table

###############读取数据################
zhengqi_train,zhengqi_test,end_train,epoches,pca = read_table()
train_min = zhengqi_train.min()
train_max = zhengqi_train.max()
zhengqi_train = (zhengqi_train-train_min)/(train_max-train_min)
X = zhengqi_train.iloc[:,:end_train]
Y = zhengqi_train.iloc[:,-1]
##################定义model#######################
model = sequential_model(end_train)
history=model.fit(X, Y, epochs=epoches,validation_split=0.2,batch_size=32,verbose=1)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='upper left')
plt.show()


