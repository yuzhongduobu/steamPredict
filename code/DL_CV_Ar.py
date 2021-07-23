import os
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,train_test_split
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.95
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#导入数据
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
shanchu=['V27']
len_shanchu = len(shanchu)
end_train = 38-len_shanchu
zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
#去属性名，留值
train_value = zhengqi_train.values
#训练集、测试集输入、输出值提取
X = train_value[:,:end_train]
Y = train_value[:,end_train]
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# define 10-fold cross validation test harness
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
for i,(train,test) in enumerate(kfold.split(X,Y)):
    model = Sequential()
    model.add(Dropout(0.2,input_dim=end_train))
    model.add(Dense(60, activation='relu', use_bias=True))
    #model.add(Dropout(0.1))

    #model.add(Dropout(0.1))

    model.add(Dense(40, activation='relu', use_bias=True))

    #model.add(Dropout(0.1))
    model.add(Dense(20, activation='tanh', use_bias=True))
    model.add(Dense(1, use_bias=True))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X[train], Y[train], epochs=100, batch_size=25, verbose=1)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names, scores))
    cvscores.append(scores)
print("交叉验证结果：平均标准差%.2f (+/- %.2f)" % (numpy.mean(cvscores), numpy.std(cvscores)))