import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def delList(dataMat):
    list = []
    for j in dataMat:
        list.append(len(j))
    #print(max(list))
    listF = []
    a = max(list)
    # x = [None]*a
    # print(x)
    i = 0
    for list in dataMat:
        newlist = []
        for j in list:
            newlist.append(j)
        i = 0
        while i < a:
            if len(newlist) >= a:
                break
            else:
                newlist.append(0)
                i = i + 1
        listF.append(newlist)
        print(len(newlist))
        # print(list)
    return listF
    #print(listF)
def loadDataSet(filename):
    df = pd.read_table(filename, sep='\t')
    return np.array(df)

def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green')
    ax.scatter(np.array(reconMat[:, 0]), reconMat[:, 1], c='red')
    plt.show()
def readCSV2List(filePath):
 try:
  file=open(filePath,'r',encoding="gbk")# 读取以utf-8
  context = file.read() # 读取成str
  list_result=context.split("\n")# 以回车符\n分割成单独的行
  #每一行的各个元素是以【,】分割的，因此可以
  length=len(list_result)
  for i in range(length):
   list_result[i]=list_result[i].split(",")
  return list_result
 except Exception :
  print("文件读取转换失败，请检查文件路径及文件编码是否正确")
 finally:
  file.close();# 操作完成一定要关闭
def read_out_file(path):
    try:
        f = open(path, 'r', encoding='utf-8')
        data = f.readlines()
        f.close()
        print("文件读取成功！")
        return data
    except IOError:
        print('文件读取失败！')
def pca(dataMat, topNfeat=999999):

    # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 2.计算样本的协方差矩阵 XXT
    covmat = np.cov(meanRemoved, rowvar=0)
    print(covmat)

    # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]    # 取前topNfeat大的特征值的索引
    redEigVects = eigVects[:, eigValInd]        # 取前topNfeat大的特征值所对应的特征向量

    # 4.将数据转换到新的低维空间中
    lowDDataMat = meanRemoved * redEigVects     # 降维之后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构数据，可在原数据维度下进行对比查看
    return np.array(lowDDataMat), np.array(reconMat)

dataMat= np.loadtxt('../word2vec/uncheckedCalls/edgUncheckedCalls.txt',dtype=np.float64)
print(dataMat)

lowDDataMat, reconMat = pca(dataMat, 1)
showData(dataMat, reconMat)
print(lowDDataMat)
print(lowDDataMat[0])
print(len(lowDDataMat))


c = [[i for i in j] for j in lowDDataMat]
with open("uncheckedCalls/uncheckedEdgCallsPCA.txt", "w") as fp:
    fp.writelines("\n".join([" ".join([str(i) for i in j]) for j in lowDDataMat]))
fp.close()
