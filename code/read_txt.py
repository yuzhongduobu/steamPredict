import pandas as pd
def read_table():
    zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
    zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')
    #shanchu=['V5','V9','V11','V14','V17','V19','V21','V22','V23','V25','V27','V32','V33','V34','V35']
    shanchu=['V5','V9','V11','V14','V17','V19','V21','V22','V23','V27','V35']
    len_shanchu = len(shanchu)
    end_train = 38-len_shanchu
    zhengqi_train.drop(labels=shanchu,axis=1,inplace=True)
    zhengqi_test.drop(labels=shanchu,axis=1,inplace=True)
    epoches = 200
    return zhengqi_train,zhengqi_test,end_train,epoches