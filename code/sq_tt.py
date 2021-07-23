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
##################训练集、验证集划分#################
X, X_test, Y, Y_test = train_test_split(zhengqi_train.iloc[:,:end_train],
                                        zhengqi_train.iloc[:,end_train],
                                        test_size=0.2,
                                        random_state=1)
##################定义model#######################
model = sequential_model(end_train)

history=model.fit(X, Y, epochs=epoches,batch_size=32,verbose=1)
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
#plt.show()

result = model.predict(X_test)

#与验证集比较
real = pd.DataFrame(Y_test,columns=['target'])
real = real.reset_index()
prediction = pd.DataFrame(result,columns=['target'])
plt.subplots(figsize=(13, 6))
fig2 = real['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
fig1 = prediction['target'].plot(subplots = True, style=['b.'])  # 画出预测结果图
#plt.show()
junfangcha = mean_squared_error(real['target'],prediction['target'])
print("均方误差(MSE): ",junfangcha)

