import pandas as pd

# 读取CSV文件
data = pd.read_csv('D:/pyProjects/pythonProject/czh/csv/smoteTT/allSet.csv')

print(data)
# 获取最后一列数据
last_column = data.iloc[:, -1]
print(last_column)

element_counts = last_column.value_counts() # 统计每个元素的出现次数
print(element_counts)

#origin 0:114 1:6040  2:500 3:139   4:129  5:36   6:177
#smote 0:6040 1:6040 2: 6040 3:6040 4:6040 5:6040 6:6040     1208
#smoteNN 0:5936 1:5675 2: 5883  3:5887  4:5000 5:4216 6:5993 1102
#smoteTT 0:6039 1:6030 2:6034 3:6037 4:5619 5:5624 6:6039    1183
