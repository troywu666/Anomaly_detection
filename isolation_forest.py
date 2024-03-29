'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-03-29 10:08:46
@LastEditors: Troy Wu
@LastEditTime: 2020-07-10 16:41:13
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.ensemble import IsolationForest

path = r'D:\troywu666\business_stuff\\'
data = pd.read_csv(path + 'data22.csv', sep = ',', index_col = 0)
sns.boxplot(data['本年EVA'])
model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
model.fit(data[['本年EVA']])
data['scores'] = model.decision_function(data[['本年EVA']])
data['anomaly'] = model.predict(data[['本年EVA']])
print(data[data['anomaly'] == -1])