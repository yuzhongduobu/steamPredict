# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:26:06 2021

@author: Administrator
"""
import os
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model,layers,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#导入数据 
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')
#先获取train_y，防止覆盖
train_y = zhengqi_train.values[:,38]
#标准化处理
zhengqi_train_mean = zhengqi_train.mean()
zhengqi_test_mean = zhengqi_test.mean()
zhengqi_train_std = zhengqi_train.std()
zhengqi_test_std = zhengqi_test.std()
zhengqi_train = (zhengqi_train-zhengqi_train_mean)/zhengqi_train_std
zhengqi_test = (zhengqi_test-zhengqi_test_mean)/zhengqi_test_std

#shanchu = ['V5','V11','V13','V14','V15','V17','V19','V20','V21','V22','V27','V28','V31']
shanchu=[]
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
zhengqi_test.drop(labels=shanchu,axis=1,inplace=True)
#去属性名，留值
train_value = zhengqi_train.values
test_value = zhengqi_test.values
#训练集、测试集输入、输出值提取
train_x = train_value[:,:end_train]
test_x = test_value[:,:end_train]
input1=train_value[:,:6]
# create model
_input1=Input(shape=(None,6),name='1')
Dense_1=Dense(6,activation='relu')(_input1)
_input2=Input(shape=(None,6),name='2')
Dense_2=Dense(6,activation='relu')(_input2)
_input3=Input(shape=(None,6),name='3')
Dense_3=Dense(6,activation='relu')(_input3)
_input4=Input(shape=(None,6),name='4')
Dense_4=Dense(6,activation='relu')(_input4)
_input5=Input(shape=(None,6),name='5')
Dense_5=Dense(6,activation='relu')(_input5)
_input6=Input(shape=(None,8),name='6')
Dense_6=Dense(8,activation='relu')(_input6)
concatenated = layers.concatenate([Dense_1,Dense_2,Dense_3,Dense_4,Dense_5,Dense_6],axis=-1)
Dense_7=Dense(38,activation='relu')(concatenated)
Dropout_1=Dropout(0.2)(Dense_7)
Dense_8=Dense(50,activation='relu')(Dropout_1)
Dense_9=Dense(30,activation='relu')(Dense_8)
#Dense_10=Dense(20,activation='relu')(Dense_9)
Dense_last=Dense(1)(Dense_9)
model = Model([_input1,_input2,_input3,_input4,_input5,_input6],Dense_last)
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])

history=model.fit([train_x[:,:6],train_x[:,6:12],train_x[:,12:18],train_x[:,18:24],train_x[:,24:30],train_x[:,30:38]],train_y, epochs=1000, batch_size=32,verbose=0)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.show()
print('训练集的均方误差是:',model.evaluate([train_x[:,:6],train_x[:,6:12],train_x[:,12:18],train_x[:,18:24],train_x[:,24:30],train_x[:,30:38]],train_y))
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='../tmp/model.png',dpi=256)
test_y = model.predict([test_x[:,:6],test_x[:,6:12],test_x[:,12:18],test_x[:,18:24],test_x[:,24:30],test_x[:,30:38]],batch_size=32)
outputfile = '../tmp/测试结果.txt'
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=7) 
numpy.savetxt(outputfile,test_y,fmt='%.07f')

#与历史最佳比较
bla = pd.read_table('../tmp/kkp.txt',encoding='utf-8')
ala = pd.DataFrame(test_y,columns=['target'])
plt.subplots(figsize=(13, 6))
fig2 = bla['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
fig1 = ala['target'].plot(subplots = True, style=['b.'])  # 画出预测结果图
plt.show()
chazhi = ala['target'] - bla['target']
junfangcha = mean_squared_error(bla['target'],ala['target'])
print("均方误差(MSE): ",junfangcha)
print("方向： ",numpy.sign(chazhi.mean()))