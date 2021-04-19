'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-10 17:26:43
@LastEditors: Troy Wu
@LastEditTime: 2020-07-10 18:25:49
'''
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
# 基于方差的鲁棒的异常检测模型，该模型假设正常样本都服从高斯分布

path = r'D:\troywu666\business_stuff\\'
data = pd.read_csv(path + 'data22.csv', sep = ',', index_col = 0)
trans_eva = LocalOutlierFactor(n_neighbors = 5).fit_predict(data[['本年EVA']])
print(data[trans_eva == -1])