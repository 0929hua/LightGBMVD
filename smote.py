from random import random

import xlwt  # 负责写excel
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

def generateExcel(matrix):
    matrix = np.array(matrix)
    filename =xlwt.Workbook() # 创建工作簿
    sheet1 = filename.add_sheet(u'sheet1',cell_overwrite_ok=True) # 创建sheet
    [h,l] = matrix.shape # h为行数，l为列数
    for i in range (h):
        for j in range (l):
            sheet1.write(i,j,str(matrix[i,j]))
    filename.save('data.xlsx') #保存到当前工作目录

def savaToExcel(data_resampled):
    '''
    将dataframe存储到excel中
    '''
    outputPath = 'D:/pyProjects/pythonProject/czh/csv/smote/allSmote.xlsx'
    writer = pd.ExcelWriter(outputPath)
    data_resampled.to_excel(writer)
    writer.close() # 关闭文件


if __name__ == '__main__':
    # generateExcel(matrix)
    col_name = ['特征1', '特征2', '特征3','特征4', '专业类别']
    dataset = pd.read_csv('D:/pyProjects/pythonProject/czh/csv/origin/all.csv', engine='python')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # 核心代码
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    data_resampled = np.zeros([len(y_resampled), 5]) # 构造 40 x 7 的元素为0的矩阵
    # 合并数据（SMOTE过采样后的数据：少数、多数的样本都是20，共40个）
    data_resampled[:, :-1] = X_resampled
    data_resampled[:, -1] = y_resampled

    # 将数据保存成dataframe，为了后面保存到Excel中
    data_resampled = pd.DataFrame(data_resampled, columns=col_name)
    print(data_resampled)

    # 保存数据至excel
    savaToExcel(data_resampled)


