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
X = zhengqi_train.iloc[:,:end_train]
Y = zhengqi_train.iloc[:,-1]
##################定义model#######################
model = sequential_model(end_train)
history=model.fit(X, Y, epochs=epoches,batch_size=32,verbose=1)
model.save('../tmp/model/steam.h5')

result = model.predict(zhengqi_test)

outputfile = '../tmp/result.txt'
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=3)
numpy.savetxt(outputfile,result,fmt='%.03f')

#与历史最佳比较
bla = pd.read_table('../tmp/kkp.txt',encoding='utf-8')
ala = pd.DataFrame(result,columns=['target'])
plt.subplots(figsize=(13, 6))
fig2 = bla['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
fig1 = ala['target'].plot(subplots = True, style=['b.'])  # 画出预测结果图
plt.show()
chazhi = ala['target'] - bla['target']
junfangcha = mean_squared_error(bla['target'],ala['target'])
print("均方误差(MSE): ",junfangcha)
print("方向： ",numpy.sign(chazhi.mean()))

