import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
zhengqi_train = pd.read_table('../data/zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('../data/zhengqi_test.txt',encoding='utf-8')

columns =  zhengqi_train.columns.values.tolist()
# for column in columns[0:38]:
#    g = sns.kdeplot(zhengqi_train[column], color="Red", shade = True)
#    g = sns.kdeplot(zhengqi_test[column], ax =g, color="Blue", shade= True)
#    g.set_xlabel(column)
#    g.set_ylabel("Frequency")
#    g = g.legend(["train","test"])
#    plt.show()
g = sns.kdeplot(zhengqi_train[columns[38]], color="Red", shade = True)
g.set_xlabel("target")
g.set_ylabel("Frequency")
plt.show()
