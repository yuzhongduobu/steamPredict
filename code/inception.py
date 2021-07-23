#inception validation_split
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
###############读取数据、去特征################
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')
#shanchu=['V5','V14','V17','V21','V22','V25','V27','V32','V33','V34']、
shanchu=[]
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)

##################标准化###########################
# train_std=zhengqi_train.std()
# train_mean=zhengqi_train.mean()
# zhengqi_train = (zhengqi_train-train_mean)/train_std
##################归一化###########################
#训练集
train_min=zhengqi_train.min()
train_max=zhengqi_train.max()
zhengqi_train=(zhengqi_train-train_min)/(train_max-train_min)
#测试集
test_min=zhengqi_test.min()
test_max=zhengqi_test.max()
zhengqi_test=(zhengqi_test-test_min)/(test_max-test_min)
############训练集、测试集输入、输出值提取#######
X = zhengqi_train.iloc[:,:end_train]
Y = zhengqi_train.iloc[:,end_train]
#inception结构
_input1=Input(shape=[None,6])
Dense_1=Dense(6,activation='relu')(_input1)

_input2=Input(shape=[None,6])
Dense_2=Dense(6,activation='relu')(_input2)

_input3=Input(shape=[None,6])
Dense_3=Dense(6,activation='relu')(_input3)

_input4=Input(shape=[None,6])
Dense_4=Dense(6,activation='relu')(_input4)

_input5=Input(shape=[None,6])
Dense_5=Dense(6,activation='relu')(_input5)

_input6=Input(shape=[None,8])
Dense_6=Dense(8,activation='relu')(_input6)

#融合层
concatenated = layers.concatenate([Dense_1,Dense_2,Dense_3,Dense_4,Dense_5,Dense_6],axis=-1)
#Dropout_6=Dropout(0.2)(concatenated)
Dense_7=Dense(50,activation='relu')(concatenated)
Dense_8=Dense(50,activation='relu')(Dense_7)
Dense_last=Dense(1)(Dense_8)
model = Model([_input1,_input2,_input3,_input4,_input5,_input6],Dense_last)
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
##################model fit#######################
#for i in range(6):
history=model.fit([X.iloc[:,:6],X.iloc[:,6:12],X.iloc[:,12:18],X.iloc[:,18:24],X.iloc[:,24:30],X.iloc[:,30:38]], Y, epochs=500,validation_split=0.2,batch_size=32,verbose=1)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.show()
##################model predict#######################
# result = model.predict([zhengqi_test.iloc[:,:6],zhengqi_test.iloc[:,6:12],zhengqi_test.iloc[:,12:18],zhengqi_test.iloc[:,18:24],zhengqi_test.iloc[:,24:30],zhengqi_test.iloc[:,30:38]])
# #反归一化
# result = result*(train_max['target']-train_min['target'])+train_min['target']
# outputfile = '../tmp/result.txt'
# numpy.set_printoptions(suppress=True)
# numpy.set_printoptions(precision=7)
# numpy.savetxt(outputfile,result,fmt='%.07f')
#
# #与历史最佳比较
# bla = pd.read_table('../tmp/kkp.txt',encoding='utf-8')
# ala = pd.DataFrame(result,columns=['target'])
# plt.subplots(figsize=(13, 6))
# fig2 = bla['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
# fig1 = ala['target'].plot(subplots = True, style=['b.'])  # 画出预测结果图
# plt.show()
# chazhi = ala['target'] - bla['target']
# junfangcha = mean_squared_error(bla['target'],ala['target'])
# print("均方误差(MSE): ",junfangcha)
# print("方向： ",numpy.sign(chazhi.mean()))