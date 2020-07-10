'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-10 15:28:15
@LastEditors: Troy Wu
@LastEditTime: 2020-07-10 15:36:22
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = r'D:\troywu666\business_stuff\\'
data = pd.read_csv(path + 'data22.csv', sep = ',', index_col = 0)
trans_eva = StandardScaler().fit_transform(data[['本年EVA']])
data[(trans_eva > 3) | (trans_eva < -3)]
# 相当于使用3倍标准差来找出异常点