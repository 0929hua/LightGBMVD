import json

import numpy as np
import pandas as pd
# 打开json文件
with open("results_wild.json") as f:
    data = json.load(f)

dataAdress= pd.read_csv('nb_liness.csv').values.flatten()
print(dataAdress)
count=0
labelList=[]
for adress in data:
    x=[]
    x.append(adress)
    label = []
#adress="0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208"
    dataNew=data[adress]['tools']
    # 获取第一层键值对
    for key1 in dataNew:
        #print(key1)
        # 获取第二层键值对
        for key2 in dataNew[key1]:
            if str(key2).__eq__('categories'):
                dict=dataNew[key1][key2]
                result = str(dict)
                print(result)
                if result.__contains__('access_control'):
                    label.append(0)
                if result.__contains__('arithmetic'):
                    label.append(1)
                if result.__contains__('denial_service'):
                    label.append(2)
                if result.__contains__('front_running'):
                    label.append(3)
                if result.__contains__('reentrancy'):
                    label.append(4)
                if result.__contains__('time_manipulation'):
                    label.append(5)
                if result.__contains__('unchecked_low_calls'):
                    label.append(6)
                if result.__contains__('Other'):
                    label.append(7)
                if result.__eq__(' '):
                    label.append('null')
    label = list(set(label))
    x.append(label)
    labelList.append(x)
    count = count + 1
    print("正在遍历中。。。")
    print(count)
print(labelList)

file_name = "m.txt"  # TXT文件名称
list0=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
for i in labelList:
    s=str(i)
    if s.endswith(', [0]]'):
        list0.append(i)
    if s.endswith(', [1]]'):
        list1.append(i)
    if s.endswith(', [2]]'):
        list2.append(i)
    if s.endswith(', [3]]'):
        list3.append(i)
    if s.endswith(', [4]]'):
        list4.append(i)
    if s.endswith(', [5]]'):
        list5.append(i)
    if s.endswith(', [6]]'):
        list6.append(i)

file_name = "m2.txt"  # TXT文件名称
# 打开或新建TXT文件进行写入操作
with open(file_name, 'w') as file:
    for item in list0:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list1:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list2:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list3:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list4:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list5:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
    for item in list6:
        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
print("已将列表保存到", file_name)
#7151


###
#access_control 0 115                    114   * 0.2 = 22
#arithmetic 1     6049  2603~8651 -10    6040  * 0.2 = 1208
#denial_service 2 505   8652~9156 -5     500   * 0.2 = 100
#front_running 3  139   9645~9783        139   * 0.2 = 27
#reentrancy 4     129   10097~10225      129   * 0.2 = 25
#time_manipulation 5  36  10288~10323    36    * 0.2 = 7
#unchecked_low_calls 6 178  177          177   * 0.2 = 35

