import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm.sklearn import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from joblib import dump, load
import numpy
def acc(y_test,test_predict):
    Accuracy = accuracy_score(y_test, test_predict)
    print(Accuracy)

    precision = precision_score(y_test, test_predict, average='micro')
    print(precision)

    recall = recall_score(y_test, test_predict, average='micro')
    print(recall)

    f1_micro = f1_score(y_test, test_predict, average='micro')
    print(f1_micro)

    f1_macro = f1_score(y_test, test_predict, average='macro')
    print(f1_macro)

def train(X_train, X_test, y_train, y_test):
    clf_multilabel = OneVsRestClassifier(LGBMClassifier(n_estimators=1000))

    model = clf_multilabel.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(pred)
    # 模型评估
    # error_rate=np.sum(pred!=test.lable)/test.lable.shape[0]
    error_rate = np.sum(pred != y_test) / y_test.shape[0]
    print('测试集错误率(softmax):{}'.format(error_rate))

    accuray = 1 - error_rate
    print('测试集准确率：%.4f' % accuray)
    # 模型保存
    dump(model, 'lightGBMSmoteTT.joblib')

#读入数据
dataset = pd.read_csv('../../csv/smoteTT/TrainSET.csv', engine='python')

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=123)

# #模型训练
# train(X_train, X_test, y_train, y_test)

# 模型预测
data = numpy.loadtxt(open("../../csv/smoteTT/PredictSET.CSV","rb"),delimiter=",",skiprows=0)

label=numpy.loadtxt(open("../../csv/smoteTT/PredictSETLabel.csv","rb"),delimiter=",",skiprows=0)

x_pred = np.array(data)
y_true=np.array(label)

#加载模型
clf = load('lightGBMSmoteTT.joblib')
y_pred = [round(value) for value in clf.predict(x_pred)]
print('y_pred：', y_pred)
acc(y_true, y_pred)
# 0.9735539186088636
# 0.9735539186088636
# 0.9735539186088636
# 0.9735539186088636
# 0.973522650782215
x=6
#1  # 0：1692 1：0  2：3  3：5  4：0  5 ：0  6:111
label1=y_pred[0:1183]
print(label1)
m1=0
for i in label1:
    if i==x:
        m1=m1+1
print(m1)

#1  # 0：1 1：1676  2：6  3：9 4：4  5 ：1  6:111
label2=y_pred[1183:2366]
print(label2)
m2=0
for i in label2:
    if i==x:
        m2=m2+1
print(m2)

#1  # 0：2 1：9  2：1688  3：1 4：0  5 ：1  6:111
label3=y_pred[2366:3549]
print(label3)
m3=0
for i in label3:
    if i==x:
        m3=m3+1
print(m3)

#1  # 0：4 1：3  2：5  3：1677 4：8  5 ：0  6:111
label4=y_pred[3549:4732]
print(label4)
m4=0
for i in label4:
    if i==x:
        m4=m4+1
print(m4)

#1  # 0：7 1：7  2：5  3：5  4：1677  5 ：0  6:111
label5=y_pred[4732:5915]
print(label5)
m5=0
for i in label5:
    if i==x:
        m5=m5+1
print(m5)

#1  # 0：1 1：1  2：2  3：0  4：1  5 ：1695  6:111
label6=y_pred[5915:7098]
print(label6)
m6=0
for i in label6:
    if i==x:
        m6=m6+1
print(m6)

#1  # 0：0 1：2  2：1  3：4  4：0  5 ：0  6:111
label7=y_pred[7098:8281]
print(label7)
m7=0
for i in label7:
    if i==x:
        m7=m7+1
print(m7)

# # #1183
Tp1=1172
Tp2=1162
Tp3=1171
Tp4=1172
Tp5=1062
Tp6=1142
Tp7=1181
#
Fn1=11
Fn2=21
Fn3=12
Fn4=11
Fn5=121
Fn6=41
Fn7=2
#
Fp1=10
Fp2=8
Fp3=21
Fp4=18
Fp5=32
Fp6=122
Fp7=8

tp=Tp1+Tp2+Tp3+Tp4+Tp5+Tp6+Tp7
fp=Fp1+Fp2+Fp3+Fp4+Fp5+Fp6+Fp7
fn=Fn1+Fn2+Fn3+Fn4+Fn5+Fn6+Fn7
sum=2*tp+fp+fn
micro1=2*tp/sum
print("micro1: " + str(micro1))

sum1=2*Tp1+Fp1+Fn1
f1=(2*Tp1)/sum1
print("f1: ")
print(f1)

sum2=2*Tp2+Fp2+Fn2
f2=(2*Tp2)/sum2
print("f2: ")
print(f2)

sum3=2*Tp3+Fp3+Fn3
f3=(2*Tp3)/sum3
print("f3: ")
print(f3)

sum4=2*Tp4+Fp4+Fn4
f4=(2*Tp4)/sum4
print("f4: ")
print(f4)

sum5=2*Tp5+Fp5+Fn5
f5=(2*Tp5)/sum5
print("f5: ")
print(f5)

sum6=2*Tp6+Fp6+Fn6
f6=(2*Tp6)/sum6
print("f6: ")
print(f6)

sum7=2*Tp7+Fp7+Fn7
f7=(2*Tp7)/sum7
print("f7: ")
print(f7)

macro1=(f1+f2+f3+f4+f5+f6+f7)/7
print("macro1: " +str(macro1))

# micro1: 0.9735539186088636
# f1:
# 0.9911205073995771
# f2:
# 0.9876753081172971
# f3:
# 0.9861052631578947
# f4:
# 0.987779182469448
# f5:
# 0.932806324110672
# f6:
# 0.93338782182264
# f7:
# 0.9957841483979764
# macro1: 0.973522650782215