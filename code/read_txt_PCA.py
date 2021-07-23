from sklearn.decomposition import PCA
import os
import numpy
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sequential import sequential_model
from read_txt import read_table
def read_table():
    end_train = 15
    zhengqi_train = pd.read_table('../data/zhengqi_train.txt', encoding='utf-8')
    zhengqi_test = pd.read_table('../data/zhengqi_test.txt', encoding='utf-8')
    pca = PCA(n_components=end_train)
    pca.fit(zhengqi_train.iloc[:,:-1])
    pca_result = pca.transform(zhengqi_train.iloc[:,:-1])
    zhengqi_test = pca.transform(zhengqi_test)
    zhengqi_test = pd.DataFrame(zhengqi_test)
    pca_result = pd.DataFrame(pca_result)
    pca_result['target']= zhengqi_train.iloc[:,-1]
    epoches = 200
    return pca_result,zhengqi_test,end_train,epoches,pca
read_table()