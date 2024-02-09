import numpy as np
from matplotlib import pyplot as plt

import csv
import json
#buggy_1  无漏洞合约 nothing false
#buggy_2  TX合约    true
#buggy_3  DAO合约   nothing  true
#buggy_4  DAO合约   nothing  true
#buggy_5  TX合约    true
#buggy_6  TX合约    true
#buggy_7  无漏洞合约 nothing false
#buggy_8  DAO合约   nothing  true
#buggy_9  DAO合约   nothing  true
#buggy_10  DAO合约   nothing  true
#buggy_11  无漏洞合约  nothing false
#buggy_12  DAO合约  nothing  true      true true
#buggy_13  DAO合约  nothing  true
#tx_1 TX合约        true
#tx_2 TX合约        true
#tx_3 TX合约        true
#tx_4 TX合约        true
#tx_5 TX合约        true     true  true
import os

#  dao 6  无漏洞3个   TX 7个  bothTwo 2个


from os.path import join, getsize
###读取文件
import os
import sys
sys.setrecursionlimit(1000000000) #例如这里设置为十万
class TailRecurseException(BaseException):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated5
    function recurses in a non-tail context.
    """
    def func(*args, **kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs

    func.__doc__ = g.__doc__
    return func


def read_out_file(path):
 try:
  f = open(path, 'r', encoding='utf-8')
  data = f.readlines()
  f.close()
  print("文件读取成功！")
  return data
 except IOError:
  print('文件读取失败！')
def get_filelist(dir, path, format):
    # 存放对应Json文件
    Filelist = [];
    for home, dir, files, in os.walk(path):
        # 遍历对应的文件下的所有文件
        for filename in files:
            # 通过文件后缀判断是否为format文件
            if filename.endswith(format):
                Filelist.append(os.path.join(home, filename))
    return Filelist
###提取block中 操作码
def operateCodeList(data):
    #path = '../skip/decompile_dao_hack.dasm.output'
    #data = read_out_file(path)  # 读取文件
    nodeS = []  # 去除以B,[,E,S,P,F,=,\n开头的字符串,包含 : 的字符串
    for i in data:
        if i.startswith('B') or i.startswith('[') or i.startswith('E') or i.startswith('S') \
                or i.startswith('P') or i.startswith('F') or i.startswith('=') or i.startswith('\n') or \
                i.__contains__(':'):
            continue
        m = i.lstrip().strip('\n')  # 去除首尾空白字符
        nodeS.append(m)
    # print(nodeS)
    codelist = []
    codeLists = []
    for opcode in nodeS:
        if opcode.__contains__("---"):
            codeLists.append(codelist)
            codelist = []
            continue
        codelist.append(opcode)
    # print(codeLists)

    operateCodeList = []  # 区块节点操作码集合
    for i in codeLists:
        if i.__eq__([]):  # 去除空集
            continue
        operateCodeList.append(i)
    return operateCodeList

#blockList  区块节点
#preSucList   区块节点的后继节点列表
def blockListAndpreSucListS(data):
    blockListAndpreSucList=[]
    blockList = []  # 区块节点
    preSucList = []  # 区块节点的后继节点列表
    succList=[]# 区块节点的前驱节点列表
    for i in data:
        if i.startswith('Block'):
            m = i.lstrip().strip('\n')
            blockList.append(m)
        if i.startswith('Successors'):  # i.startswith('Predecessors') or
            preSucList.append(i)
        if i.startswith('Predecessors'):  # i.startswith('Predecessors') or
            succList.append(i)
        blockListS = []
    #print(blockList)
    blockListS = []
    for i in blockList:
        blockListS.append(i.split()[1])  # 以空格为分隔符，包含 \n
    preSucListS = []
    for i in preSucList:
        list = i.split(':')[1]  # 以空格为分隔符，包含 \n
        preSucListS.append(list)
    succListS=[]
    for i in succList:
        list = i.split(':')[1]  # 以空格为分隔符，包含 \n
        succListS.append(list)
    blockListAndpreSucList.append(blockListS)
    blockListAndpreSucList.append(preSucListS)
    blockListAndpreSucList.append(succListS)
    return blockListAndpreSucList

def Successors(blockNumAndSuccessors):
    nodeAll = []
    for i in blockNumAndSuccessors:
        str = i[1]
        if (str == ' '):
            node = []
            nodeAll.append(node)
        else:
            r = (str.replace('[', '').replace(']', '')).split(',')
            node = []
            for j in r:
                m = j.lstrip().strip('\n')  # 去除首尾空白字符
                node.append(m)
            nodeAll.append(node)
    return nodeAll

###深度优先算法
def dfsTravel(graph, source):
    # 传入的参数为邻接表存储的图和一个开始遍历的源节点
    travel = []  # 存放访问过的节点的列表
    stack = [source]  # 构造一个堆栈
    while stack:  # 堆栈空时结束
        current = stack.pop()  # 堆顶出队
        if current not in travel:  # 判断当前结点是否被访问过
            travel.append(current)  # 如果没有访问过，则将其加入访问列表
        for next_adj in graph[current]:  # 遍历当前结点的下一级
            if (next_adj==''):
                continue
            if next_adj not in travel:  # 没有访问过的全部入栈
                stack.append(next_adj)
    return travel

# 区块号以及区块节点的后继节点列表
#blockList  区块节点
#preSucList   区块节点的后继节点列表
def  blockNumAndSuccessors(blockListS,preSucListS):
    blockNumAndSuccessors = []  # 区块号以及区块节点的后继节点列表
    m = 0
    f=len(blockListS)+1
    for i in blockListS:
        nodeList = []
        nodeList.append(i)
        n = 0
        for j in preSucListS:
            if m == n:
                j=j.replace("\n", "")
                s = j.strip(' [')
                s = s.strip(']')
                if s.__eq__(''):
                    nodeList.append(str(f))
                else:
                    nodeList.append(s)
            n = n + 1
        m = m + 1
        blockNumAndSuccessors.append(nodeList)
    return blockNumAndSuccessors

###将节点以及其后继节点的操作码指令
def blockOperS(numCodeDictS,codeOperDict,blockListS):
    newDict={}
    Mnodes=[]
    for key ,value in numCodeDictS.items():
        #print(key, ":", value)
        Mnode = []
        for i in value:
            for key1, value1 in codeOperDict.items():
                if i.__eq__(key1):
                    Mnode.append(value1)
                    Mnodes.append(Mnode)
    #print(Mnodes)
    res = []
    for i in Mnodes:
        if i not in res:
            res.append(i)
    newDict = {}  # 将节点以及其后继节点的操作码指令  转换为 字典格式
    for i, j in zip(blockListS, res):  # 同时循环两个列表
        newDict[i] = j  # 此时i为键，j为值，即{i：j}

    return newDict
def getFileName(dir,path,format):
    Filelist = [];
    for home, dir, files, in os.walk(path):
        # 遍历对应的文件下的所有文件
        for filename in files:
            if filename.endswith(format):
                Filelist.append(filename.strip(format))
    return Filelist
###提取控制流图所有执行路径
#fileExtractPath=''
#@tail_call_optimized
def add(key,dict,newKey):
    global fileExtractPath
    newKey=newKey+'->'+key;
    if not key=='':
        list = dict[key]
        if len(list)==1 & list[0].__eq__(''):
                 fileExtractPath.write(newKey+'\n')
                 print(newKey)
                 return 'true'
        for i in list:
            add(i, dict, newKey)
###提取所有可执行路径
def extractPath(numCodeDictS,extractPath):
    pathS = []  # 提取控制流图中所有执行路径
    f = open(extractPath, 'r')
    for line in f.readlines():
        curLine = line.strip('\n')
        if curLine.startswith('->'):
            m = curLine.strip('->')
        pathS.append(m)
    #print(pathS)
    pathALL = []
    for path in pathS:
        list = path.split('->')
        pathALL.append(list)
    return pathALL
fileExtractPath=''
def addC(key, dict, newKey):
    newKey = newKey + '->' + key;
    if not key == '':
        list = dict[key]
        if len(list) == 1 & list[0].__eq__(''):
            fileExtractPath.write(newKey + '\n')
            #print(newKey)
            return
        for i in list:
            if newKey.__contains__(i):
                fileExtractPath.write(newKey + '\n')
                #print(newKey)
                i = ''
            addC(i, dict, newKey)
#去除重复路径
def removeRepeat(data):
    m = len(data) - 1
    while m >= 0:
        strS = data[m]
        strS = strS.strip('\n')
        str = data[m - 1]
        str = str.strip('\n')
        if str in strS:
            data.remove(str + '\n')
        m = m - 1
    return data

flag=''
def daoCheck(pathALL,codeOperDict):
    flag = 'none'  # 默认表示无漏洞
    listCall = []
    for path in pathALL:
        for i in path:
            for key, value in codeOperDict.items():
                if i.__eq__(key):
                    list = []
                    # print(value)
                    for code in value:
                        code = code.split(' ')[1]
                        list.append(code)
                    if list.__contains__('CALL') & list.__contains__('CALLER'):
                        if   list.__contains__('RETURNDATASIZE') :
                            continue
                        else:
                            listCall.append(value)

    if len(listCall)>0:
        key = 'SSTORE'
        for codeList in listCall:
            opList = []
            locationNum = []
            dict = {}
            i = 0
            keyNum = 0
            for code in codeList:
                if code.__contains__(key):
                    keyNum = keyNum + 1
            if keyNum == 0:
                flag = 'true'
                break
            if flag.__eq__('none'):
                for code in codeList:
                    if code.__contains__('SHA3'):
                        opList.append('SHA3')
                        i = i + 1
                        locationNum.append(i)
                    if code.__contains__('SLOAD'):
                        opList.append('SLOAD')
                        i = i + 1
                        locationNum.append(i)
                    if code.__contains__('SSTORE'):
                        opList.append('SSTORE')
                        i = i + 1
                        locationNum.append(i)
                    if code.__contains__('GAS'):
                        opList.append('GAS')
                        i = i + 1
                        locationNum.append(i)
                    if code.__contains__('CALL'):
                        opList.append('CALL')
                        i = i + 1
                        locationNum.append(i)
                for i, j in zip(opList, locationNum):  # 同时循环两个列表
                    dict[i] = j  # 此时i为键，j为值，即{i：j}
                # print("dict:")
                # print(dict)
                sstore = dict['SSTORE']
                call = dict['CALL']
                if sstore < call:
                    flag = 'false'
                else:
                    flag = 'true'
    else:
        flag = 'none'
    return flag
def txOriginCheck(pathALL,codeOperDict):
    flag = 'none'  # 默认表示无漏洞
    listTX = []
    for path in pathALL:
        for i in path:
            for key, value in codeOperDict.items():
                if i.__eq__(key):
                    list = []
                    # print(value)
                    for code in value:
                        code = code.split(' ')[1]
                        list.append(code)
                    if list.__contains__('ORIGIN') & list.__contains__('EQ') :
                        #if not list.__contains__('CALL'):
                            listTX.append(value)
    print('len(listTX)')
    print(len(listTX))
    if len(listTX) > 0:
        for codeList in listTX:
            i = 0
            locationNum = []
            newList = []
            dict = {}
            for code in codeList:
                if code.__contains__('ORIGIN'):
                    i = i + 1
                    locationNum.append(i)
                    newList.append('ORIGIN')
                if code.__contains__('EQ'):
                    i = i + 1
                    locationNum.append(i)
                    newList.append('EQ')
            for i, j in zip(newList, locationNum):  # 同时循环两个列表
                dict[i] = j  # 此时i为键，j为值，即{i：j}
            ORIGIN = dict['ORIGIN']
            EQ = dict['EQ']
            if ORIGIN < EQ:
                flag = 'true'
                break
            else:
                flag = 'false'
        print('listTX')
        #print(listTX)
    else:
        flag = 'none'
    return flag
def exchange(file_path, result_dict):
    # 指定要操作的txt文件路径
    # 打开txt文件
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 逐行读取所有内容
    j = 0
    for m in first_column:
        value = result_dict.get(m)
        # 对每一行进行处理
        for i in range(len(lines)):
            line = lines[i]
            # print(line)
            if m in line:  # 判断该行是否包含固定语句
                new_line = line.replace(m, str(value))  # 替换为新的语句
                lines[i] = new_line  # 更新原始列表中的元素

    # 保存修改后的结果到同名文件（会覆盖原文件）
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line)
def change(file_path):
    # 打开txt文件
    file = open(file_path, "r")
    lines = file.readlines()  # 逐行读取内容到列表中
    file.close()  # 关闭文件

    # 去除每行结尾的换行符（\n）
    lines_without_newline = [line[:-1] for line in lines]

    # 将字符串类型的元素转换为int或float等需要的数据类型
    array = []
    for line in lines_without_newline:
        array.append(eval(line))  # eval函数会根据输入进行动态求值
    return array
def convert_to_adjacency_matrix(predecessor):
    num_nodes = len(predecessor) + 1  # 加上根节点

    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    for i in range(len(predecessor)):
        start_node = predecessor[i][0] - 1  # 索引从0开始计算，因此-1
        end_node = predecessor[i][1] - 1  # 索引从0开始计算，因此-1

        adjacency_matrix[start_node][end_node] = 1
        adjacency_matrix[end_node][start_node] = 1

    return adjacency_matrix
def compare_length(s):
        return len(s)

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
def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green')
    ax.scatter(np.array(reconMat[:, 0]), reconMat[:, 1], c='red')
    plt.show()
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

import time

if __name__ == "__main__":
    begin_time = time.time()
    print(begin_time)
    conOpCodesPath = 'D:/opcode/reentrancy/'
    conOpCodesPathList = get_filelist(dir, conOpCodesPath, "txt")
    #print(conOpCodesPathList)
    daoNum=0
    num=0
    txNum=0
    normal=0
    bothTwo=0
    daoList=[]
    txList=[]
    listLast = []
    label=[]
    for file in conOpCodesPathList:
      #if file.__eq__('D:/opcode/access/4.txt'):
            #if getsize(file)!= 855:
                num = num + 1
                #print(getsize(file))
                flag = 'false'  # 默认表示无漏洞
                data = read_out_file(file)  # 读取文件
                blockListAndpreSucList = blockListAndpreSucListS(data)
                blockListS = blockListAndpreSucList[0]  # 区块节点
                preSucListS = blockListAndpreSucList[1]  # 区块节点的后继节点列表
                succListS = blockListAndpreSucList[2]  # 区块节点的前继节点列表
                blockNumAndSuccessor = blockNumAndSuccessors(blockListS, preSucListS)  # 区块号以及区块节点的后继节点列表
                blockNumAndprecessor = blockNumAndSuccessors(blockListS, succListS)  # 区块号以及区块节点的前继节点列表
                Successor = Successors(blockNumAndSuccessor)  # 后继节点


                print(blockNumAndprecessor)

                # 获取多维数组的第一列
                first_column = []
                for row in blockNumAndprecessor:
                    first_column.append(row[0])
                first_column = sorted(first_column, key=lambda x: len(x), reverse=True)
                print(first_column)

                array = [i for i in range(1, len(first_column) + 1)]
                print(array)
                reversed_arr = array[::-1]
                result_dict = dict(zip(first_column, reversed_arr))  # 使用zip函数将两个数组转换为字典
                print(result_dict)  # 输出结果

                with open('../json.txt', 'w') as file:
                    for item in blockNumAndprecessor:
                        file.write(str(item) + '\n')  # 每次写入一项后加上换行符'\n'
                exchange('../json.txt', result_dict)
                m = change('../json.txt')
                print(m)
                p=[]
                for i in m:
                    k=[]
                    for j in i:
                        if len(j)<=2:
                            j=int(j)
                            k.append(j)
                        else:
                            result_list = j.split(",")
                            s=[]
                            for i in result_list:
                                i=i.strip(' ')
                                s.append(int(i))
                            for c in s:
                                k.append(c)
                    p.append(k)            #print(result_list)
                print(p)
                print(len(p))
                a = [tuple(i) for i in p]
                print(a)
                adjacency_matrix = convert_to_adjacency_matrix(a)
                print(adjacency_matrix)
                values, vectors = np.linalg.eig(adjacency_matrix)
                print(vectors)
                r=[]
                for i in vectors:
                    m=i.sum()
                    m=float(m)
                    r.append(m)
                print(r)
                print(len(r))
                label.append(r)
    data = delList(label)
    dataMat = np.array(data)
    # ndarray转为list
    b = dataMat.tolist()
    # 二维数组遍历，列表生成式
    c = [[i for i in j] for j in b]
    with open("reentrancy/edgReentrancy.txt", "w") as fp:
        fp.writelines("\n".join([" ".join([str(i) for i in j]) for j in b]))
    fp.close()









