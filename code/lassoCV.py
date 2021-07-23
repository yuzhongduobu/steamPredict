# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:14:41 2021

@author: Administrator
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso,LassoCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR,SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# 导入数据
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')
####################删除训练集与测试集分布不同的属性#############################
shanchu = ['V5','V9','V11','V14','V17','V19','V20','V21','V22','V28'] #V9 V11 V14 V19 V20 V21 V27待定
#shanchu=[]
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
zhengqi_test.drop(labels=shanchu,axis=1,inplace=True)
################ 数据拆分 训练集 测试集 #################
#zhengqi_train_train,zhengqi_train_test = train_test_split(zhengqi_train,test_size=0.01,random_state=10)
#print(zhengqi_train.info())
#print(zhengqi_train_train.info())
#print(zhengqi_train_test)
'''
################ 数据分析 ###########################
# 描述性统计分析
description = [zhengqi_train.min(), zhengqi_train.max(), zhengqi_train.mean(), zhengqi_train.std()]  # 依次计算最小值、最大值、均值、标准差
description = pd.DataFrame(description, index = ['Min', 'Max', 'Mean', 'STD']).T  # 将结果存入数据框
print('描述性统计结果：\n',np.round(description, 2))  # 保留两位小数
print("=========================")
print(zhengqi_train.info())
'''
'''
################## 相关性分析 #########################
# 相关性分析
corr = zhengqi_train.corr(method = 'pearson')  # 计算相关系数矩阵
print('相关系数矩阵为：\n',np.round(corr, 2))  # 保留两位小数

# 绘制热力图
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(40, 40)) # 设置画面大小 
sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Reds") 
plt.title('相关性热力图')
plt.show()
plt.close
'''

################## Lasso选取关键变量 ###############

###lassoCV选择########
lasso = LassoCV(cv=3,random_state=0)
lasso.fit(zhengqi_train.iloc[:,0:end_train],zhengqi_train['target'])
print('lassoCV评分：',lasso.score(zhengqi_train.iloc[:,0:end_train],zhengqi_train['target']))
#print('相关系数为： ',np.round(lasso.coef_,5))
print('相关系数非零个数为： ',np.sum(lasso.coef_!=0))
mask = lasso.coef_ != 0  # 返回一个相关系数是否为零的布尔数组
#print('相关系数是否为零：',mask)

#outputfile ='../tmp/new_reg_data.csv'  # 输出的数据文件
new_reg_data = zhengqi_train.iloc[:, mask]  # 返回相关系数非零的数据
#new_reg_data.to_csv(outputfile)  # 存储数据
#print('输出数据的维度为：',new_reg_data.shape)  # 查看输出数据的维度


feature = new_reg_data.columns.values.tolist()
data_train = zhengqi_train.copy()  
data_train_mean = data_train.mean()
data_train_std = data_train.std()
#data_train = (data_train - data_train_mean)/data_train_std  # 数据标准化
x_train = data_train[feature].as_matrix()  # 属性数据
y_train = data_train['target'].as_matrix()  # 标签数据
####################线性SVR#######################################
#linearsvr = LinearSVR()  # 调用LinearSVR()函数
#linearsvr.fit(x_train,y_train)
###################核函数SVR#######################################
svr = SVR(kernel='rbf',epsilon=0.1)
svr.fit(x_train,y_train)

'''
#################### 训练文件预测 ####################################
x = ((data_train[feature] - data_train_mean[feature])/data_train_std[feature]).as_matrix()  # 预测，并还原结果。
data_train[u'target_pred'] = svr.predict(x) * data_train_std['target'] + data_train_mean['target']
outputfile = '../tmp/train_result.xls'  # SVR预测后保存的结果
data_train.to_excel(outputfile)

#print('真实值与预测值分别为：\n',data_train[['target','target_pred']])

#fig = data_train[['target','target_pred']].plot(subplots = True, style=['b-o','r-*'])  # 画出预测结果图
#plt.show()
print("均方误差(MSE): ",mean_squared_error(data_train['target'],data_train['target_pred'] ))
''' 



#################### 测试文件预测 ####################################
data_test = zhengqi_test.copy()  # 
data_test_mean = data_test.mean()
data_test_std = data_test.std()
#data_test = (data_test - data_test_mean)/data_test_std  # 数据标准化
x = data_test[feature].as_matrix()  # 预测，并还原结果。
data_test['target_pred'] = svr.predict(x)

outputfile = '../tmp/测试结果.txt'
data_test['target_pred'].to_csv(outputfile,sep='\t',index=False)


    #################与历史最佳评分进行比较################################
bla = pd.read_table('../tmp/kkp.txt',encoding='utf-8')
ala = data_test['target_pred']
plt.subplots(figsize=(13, 6))
fig2 = bla['target'].plot(subplots = True, style=['r.'])  # 画出预测结果图
fig1 = ala.plot(subplots = True, style=['b.'])  # 画出预测结果图
plt.show()
chazhi = ala - bla['target']
junfangcha = mean_squared_error(bla,ala)
print("均方误差(MSE): ",junfangcha)
print("方向： ",np.sign(chazhi.mean()))



