#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : analyze
Describe: 
    简单分析训练数据
Author : LH
Date : 2019/9/16
"""
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('yq.txt',sep='_!_',header=None,names=['id','label','lname','title','abstr'])
print('查看数据概况')
print(df.info())
print(df.describe())
labelx = [x for x in range(100,121,1)]

print('查看没有abstr的各个分类下的数据个数')
labelGroup = df.groupby(by=['label'],axis=0)
print(labelGroup.count())
idv = labelGroup['id'].count().values.tolist()
subv = (labelGroup['id'].count()-labelGroup['abstr'].count()).values.tolist()
plt.bar(labelx,idv,width=0.8,color='red',label='all num')
plt.bar(labelx,subv,width=0.8,color='green',label='sub num')
plt.xlabel('label id')
plt.ylabel('num count')
plt.title('all - sub')
plt.legend()
plt.show()
