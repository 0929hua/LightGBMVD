import json
from gensim.models import Word2Vec
from os.path import join, getsize
###读取文件
import os
import sys
def read_out_file(path):
 try:
  f = open(path, 'r', encoding='utf-8')
  data = f.readlines()
  f.close()
  #print("文件读取成功！")
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
def dealdata(dataADD):
    textF = open("./word2vecNew.txt", "w")
    textF.truncate()
    for str in dataADD:
        if str.startswith('DEST'):
            str = str.lstrip('DEST')
        if str.startswith('POP'):
            str = str.lstrip('POP')
        if str.startswith('JUMPI'):
            str = str.lstrip('JUMPI')
        if str.startswith('JUMP'):
            str = str.lstrip('JUMP')
        if str.startswith('ISZERO'):
            str = str.lstrip('ISZERO')
        if str.startswith('I'):
            str = str.lstrip('I')
        if str.startswith('PUSH3'):
            str = str.lstrip('PUSH3')
        if str.startswith('DUP3'):
            str = str.lstrip('DUP3')
        if str.startswith('PUSH20'):
            str = str.lstrip('PUSH20')
        if str.startswith('PUSH4'):
            str = str.lstrip('PUSH4')
        if str.startswith('SWAP2'):
            str = str.lstrip('SWAP2')
        if str.startswith('DIV'):
            str = str.lstrip('DIV')
        if str.startswith('DUP4'):
            str = str.lstrip('DUP4')
        if str.startswith('OR'):
            str = str.lstrip('OR')
        if str.startswith('LOG3'):
            str = str.lstrip('LOG3')
        if str.startswith('LOG1'):
            str = str.lstrip('LOG1')
        if str.startswith('SWAP3'):
            str = str.lstrip('SWAP3')
        if str.startswith('PUSH9'):
            str = str.lstrip('PUSH9')
        if str.startswith('MUL'):
            str = str.lstrip('MUL')
        if str.startswith('CALL'):
            str = str.lstrip('CALL')
        if str.startswith('SWAP7'):
            str = str.lstrip('SWAP7')
        if str.startswith('DATACOPY'):
            str = str.lstrip('DATACOPY')
        if str.startswith('DUP11'):
            str = str.lstrip('DUP11')
        if str.startswith('EQ'):
            str = str.lstrip('EQ')
        if str.startswith('RETURN'):
            str = str.lstrip('RETURN')
        if str.startswith('3'):
            str = str.lstrip('3')
        if str.startswith('BLOCKHASH'):
            str = str.lstrip('BLOCKHASH')
        if str.startswith('0'):
            str = str.lstrip('0')
        if str.startswith('1'):
            str = str.lstrip('1')
        if str.startswith('2'):
            str = str.lstrip('2')
        if str.startswith('4'):
            str = str.lstrip('4')
        if str.startswith('5'):
            str = str.lstrip('5')
        if str.startswith('6'):
            str = str.lstrip('6')
        if str.startswith('7'):
            str = str.lstrip('7')
        if str.startswith('8'):
            str = str.lstrip('8')
        if str.startswith('9'):
            str = str.lstrip('9')
        if str.startswith('GIN'):
            str = str.lstrip('GIN')
        if str.startswith('RESS'):
            str = str.lstrip('RESS')
        if str.startswith('ER'):
            str = str.lstrip('ER')
        if str.startswith('DATALOAD'):
            str = str.lstrip('DATALOAD')
        if str.startswith('VALUE'):
            str = str.lstrip('VALUE')
        if str.startswith('DATASIZE'):
            str = str.lstrip('DATASIZE')
        if str.startswith('ODE'):
            str = str.lstrip('ODE')
        if str.startswith('PRICE'):
            str = str.lstrip('PRICE')
        if str.startswith('OD'):
            str = str.lstrip('OD')
        if str.startswith('MOD'):
            str = str.lstrip('MOD')
        if str.startswith('LIMIT'):
            str = str.lstrip('LIMIT')
        textF.write(str)
    textF.close()
    return "true"
def addData(x):
    listADD = []
    #listADD.append(x)
    dataA = read_out_file("./word2vecNew.txt")
    for j in dataA:
        list = j.split(" ")
        newlist = []
        for i in list:
            if i.__eq__(""):
                continue;
            else:
                i = i.strip("\n")
                newlist.append(i)
        # print(newlist)
        m = 0
        for j in newlist:
            m = m + float(j)
        listADD.append(m)
    return listADD
import time
import numpy as np
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
if __name__ == "__main__":
    conOpCodesPath = 'D:/opcode/access/'
    conOpCodesPathList = get_filelist(dir, conOpCodesPath, ".txt")
    num=0
    label = []
    sList = []
    for file in conOpCodesPathList:
        #if file.__eq__('D:/pyProjects/pythonProject/MachineLearningVulnerabilityDetection/dataset/opcodes/overflow/60.txt'):
                data = read_out_file(file)  # 读取文件
                #print(file)
                num=num+1
                name = file.strip(conOpCodesPath)
                x = name.strip('.tx')
                print(name + "文件名:" + str(num))
                operateCodeListS = operateCodeList(data)  # 提取操作码指令
                listF=[]
                for opList in operateCodeListS:
                    newList = []
                    for i in opList:
                        list = i.split(' ')
                        for i in list:
                            if i.startswith('0'):
                                continue
                            else:
                                newList.append(i)
                    listF.append(newList)
                print("合约操作码片段总数："+str(len(listF)))
                sList = []
                for list in listF:
                    #print(list)
                    test_senTIME = []
                    test_senTIME.append(list)
                    model = Word2Vec(sentences=test_senTIME, vector_size=10, window=1, min_count=1, workers=1)
                    model.wv.save_word2vec_format('./word2vec.txt', binary=False)
                    data = read_out_file("./word2vec.txt")
                    text_file = open("./word2vecNew.txt", "w")
                    for i in list:
                        code = ''
                        for string in data:
                            if string.endswith("10\n"):
                                continue;
                            else:
                                if string.startswith(i):
                                    code = string.lstrip(i)
                        text_file.write(code)
                    text_file.close()
                    dataADD = read_out_file("./word2vecNew.txt")
                    dealdata(dataADD)
                    listADD = addData(x)
                    #print(listADD)
                    f = sum(listADD)
                    sList.append(f)
                #print(sList)
                #print(len(sList))
                label.append(sList)
                #print(label)
                print("label长度："+str(len(label)))
    data = delList(label)
    dataMat = np.array(data)
    # ndarray转为list
    b = dataMat.tolist()
    # 二维数组遍历，列表生成式
    c = [[i for i in j] for j in b]
    with open("access/labelAccess.txt", "w") as fp:
        fp.writelines("\n".join([" ".join([str(i) for i in j]) for j in b]))
    fp.close()