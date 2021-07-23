# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:09:10 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:14:41 2021

@author: Administrator
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR,SVR
from sklearn.metrics import mean_squared_error
# 导入数据
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')
#print(zhengqi_test.describe())
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
corr = zhengqi_test.corr(method = 'pearson')  # 计算相关系数矩阵
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


#shanchu = ['V0','V5','V9','V11','V14','V17','V19','V20','V21','V23','V22','V27','V28','V35']
shanchu = []
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
lamda = 0.0001
best_lam = 0;
best_fea = []
best_jf = 5
while lamda<0.02:
    ################## Lasso选取关键变量 ###############
    lasso = Lasso(lamda)
    lasso.fit(zhengqi_train.iloc[:,0:end_train],zhengqi_train['target'])
    
    print('相关系数非零个数为： ',np.sum(lasso.coef_!=0))
    mask = lasso.coef_ != 0  # 返回一个相关系数是否为零的布尔数组
    
    
    outputfile ='../tmp/new_reg_data.csv'  # 输出的数据文件
    new_reg_data = zhengqi_train.iloc[:, mask]  # 返回相关系数非零的数据
    
    feature = new_reg_data.columns.values.tolist()
    #feature = ['V0', 'V1', 'V2', 'V3', 'V9', 'V10', 'V12', 'V14', 'V17', 'V18', 'V24', 'V30', 'V33']
    
    data_train = zhengqi_train.copy()  # 
    data_train_mean = data_train.mean()
    data_train_std = data_train.std()
    data_train_max = data_train.max()
    data_train_min = data_train.min()
    #data_train = (data_train - data_train_mean)/data_train_std  # 数据标准化
    #data_train = (data_train - data_train_min)/(data_train_max-data_train_min)
    x_train = data_train[feature].as_matrix()  # 属性数据
    svr = LinearSVR()  # 调用LinearSVR()函数
    svr.fit(x_train,zhengqi_train['target'])
    
    zhengqi_train[u'target_pred'] = svr.predict(x_train)
    
    junfang = mean_squared_error(zhengqi_train['target'],zhengqi_train['target_pred'])
    print("均方误差(MSE): ",junfang)
    if junfang<best_jf :
        best_jf = junfang
        best_lam = lamda
        best_fea = feature
    lamda +=0.0001
    
    

    

