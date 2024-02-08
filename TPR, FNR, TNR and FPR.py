# 导入绘图模块
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
# 构建数据
def to_percent(temp, position):
 return '%1.0f'%(10*temp) + '%'

# TPR1:
# 0.9902813299232737
# FPR1:
# 0.001125319693094629

# TPR2:
# 0.9820971867007673
# FPR2:
# 0.0032736572890025577

# TPR3:
# 0.9948849104859335
# FPR3:
# 0.00020460358056265986

# TPR4:
# 0.9892583120204603
# FPR4:
# 0.003989769820971867

# TPR5:
# 0.9938618925831202
# FPR5:
# 0.001227621483375959

# TPR6:
# 0.9948849104859335
# FPR6:
# 0.001125319693094629
# 统一设置xy轴名称的字体大小
# 统一设置

Y2016 = [0.999, 0.994, 0.988, 0.997,0.989,0.991,1.000]
Y2017 = [0.996, 0.994, 0.995, 0.993,0.995,0.988,0.997]
Y2018 = [0.001, 0.006,0.012, 0.003,0.011,0.009,0.000]
Y2019 = [0.004, 0.006,  0.005, 0.007,0.005,0.012,0.003]
#labels=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
labels = ['Access Control', 'Arithmetic', 'Denial Service', 'Front Running', 'Reentrancy', 'Time Manipulation',
         'Unchecked Low Calls']
# labels = ['访问控制', '整数溢出', '拒绝服务', '交易顺序依赖', '可重入', '时间戳依赖',
#          '未检查调用返回值']

bar_width = 0.3
width = 0.1
# list2 = [94, 99.3, 94.3, 71.3]
# list3 = [6, 0.6,5.7, 28.7]
# list4 = [99.3, 99.6, 100, 100]
# list5 = [0.7, 0.4,  0, 0]
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 中文乱码的处理
plt.rcParams['font.sans-serif'] =[u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 绘图12
plt.bar(np.arange(7), Y2016,   color = 'steelblue', alpha = 0.4, width = bar_width)
plt.bar(np.arange(7), Y2018,bottom=Y2016,color = 'steelblue',alpha = 0.4,width = bar_width,hatch="xxx")

plt.bar(np.arange(7)+width+bar_width, Y2017, color = 'orange', alpha = 0.3, width = bar_width)
plt.bar(np.arange(7)+width+bar_width, Y2019 ,bottom=Y2017 ,color = 'orange', alpha = 0.3, width = bar_width,hatch="xxx")

# 添加轴标签
plt.xlabel('Six Vulnerabilities')
plt.ylabel('Detetion Rate')
# 添加刻度标签
plt.xticks(np.arange(7)+bar_width,labels,rotation=10,fontsize=8,fontweight='bold')
# 设置Y轴的刻度范围
plt.ylim([0.98, 1])
#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# 显示图例
num1=1.05
num2=0
num3=3
num4=0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
# 显示图形
fig.tight_layout()  # 调整整体空白
plt.show()
