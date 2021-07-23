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

#for i in range(6):
history=model.fit([X.iloc[:,:6],X.iloc[:,6:12],X.iloc[:,12:18],X.iloc[:,18:24],X.iloc[:,24:30],X.iloc[:,30:38]], Y, epochs=1000,batch_size=32,verbose=1)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.show()

result = model.predict([X_test.iloc[:,:6],X_test.iloc[:,6:12],X_test.iloc[:,12:18],X_test.iloc[:,18:24],X_test.iloc[:,24:30],X_test.iloc[:,30:38]])
#反归一化
result = result*(Y_max-Y_min)+Y_min

#与验证集比较
bla = pd.DataFrame(Y_test,columns=['target'])
bla = bla.reset_index()
ala = pd.DataFrame(result,columns=['target'])
plt.subplots(figsize=(13, 6))
fig2 = bla['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
fig1 = ala['target'].plot(subplots = True, style=['b.'])  # 画出预测结果图
plt.show()
chazhi = ala['target'] - bla['target']
junfangcha = mean_squared_error(bla['target'],ala['target'])
print("均方误差(MSE): ",junfangcha)
print("方向： ",numpy.sign(chazhi.mean()))
