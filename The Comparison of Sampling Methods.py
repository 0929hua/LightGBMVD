import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % float(height))




if __name__ == '__main__':

    # # LightGBM
    # l1 = [0.9558, 0.9594, 0.9736, 0.9940]
    # l2 = [0.7422, 0.9594, 0.9735, 0.9940]

    # #RF
    # l1 = [0.9614, 0.9575, 0.9720, 0.9924]
    # l2 = [0.7536, 0.9574, 0.9719, 0.9923]

    #XGBoost                                 #2
    # l1 = [0.9614, 0.9613, 0.9709, 0.9913]
    # l2 = [0.7645, 0.9611, 0.9708, 0.9913]
    # #SVM
    l1 = [0.8483, 0.5083, 0.7131, 0.7153]
    l2 = [0.1311, 0.5014, 0.7029, 0.7038]
    # #AdaBoost
    # l1 = [0.9445, 0.8058, 0.8436, 0.8316]
    # l2 = [0.6964, 0.8043, 0.8428, 0.8302]

    list1 = np.array([0.9558, 0.9594, 0.9736, 0.9940])   # 柱状图第一组数据
    list2 = np.array([0.7422, 0.9594, 0.9735, 0.9940])   # 柱状图第二组数据
    length = len(list1)
    x = np.arange(length)   # 横坐标范围
    name = ['Origin','Smote','SmoteTT','SmoteNN']
    # total_width, n = 1, 2
    # width = total_width / n
    total_width, n = 0.8, 2   # 柱状图总宽度，有几组数据
    width = total_width / n   # 单个柱状图的宽度
    x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
    x2 = x1 + width   # 第二组数据柱状图横坐标起始位置

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rc('font', family='SimHei', size=8)  # 设置中文显示，否则出现乱码！
    a = plt.bar(x1, l1, width=width,color = 'steelblue', alpha = 0.4)
    for i in range(len(x)):
        x[i] = x[i] + width+0.1
    b = plt.bar(x2, l2, width=width,color = 'orange', alpha = 0.3)
    plt.xticks(x, name)   # 用星期几替换横坐标x的值
    autolabel(a)
    autolabel(b)
    plt.xlabel('',fontsize=5)
    plt.ylim([0, 1])
    plt.tick_params(axis='x', width=0)
    plt.ylabel('Detection Rate')
    plt.title('')
    # 显示图例
    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.show()

