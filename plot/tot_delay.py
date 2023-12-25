import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

import numpy as np

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }

font4 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }
figsize = 8, 6
figure, ax = plt.subplots(figsize=figsize)
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

result = np.load('result/loss_gan.npy', allow_pickle=True)
result = result[:20000]
n = len(result)

loss_g_list = []
loss_d_list = []
for episode_result in result:
    loss_g, loss_d = zip(*episode_result)
    loss_g_list.append(np.mean(loss_g))
    loss_d_list.append(np.mean(loss_d))

result = np.load('result/p_gan.npy', allow_pickle=True)
result = result[:20000]
n = len(result)

p_fake, p_real = zip(*result)

x = np.arange(n)  # 10,20,50,100
# tick_label = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
# plt.xlabel("The Number of Users", font1)

tick_label = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
plt.xlabel("Corrupted probability p", font1)

plt.ylabel("Hits@1(%)", font1)


tick_label = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
plt.xlabel("Query Type", font1)

plt.ylabel("Hits@3(%)", font1)
x = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up', '2d', '3d', 'dp']
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y1 = [23.10, 22.23, 20.78, 32.11, 40.80, 21.39, 26.71, 22.44, 22.81, 35.27, 41.65, 39.01]
y2 = [26.19, 24.33, 22.86, 34.74, 43.44, 22.58, 28.14, 24.09, 23.37, 38.11, 45.62, 41.88]
y3 = [29.47, 26.48, 23.35, 35.02, 47.27, 22.81, 29.73, 24.70, 25.06, 40.14, 46.34, 42.39]
y4 = [30.73, 26.61, 24.13, 35.67, 48.01, 23.07, 30.11, 25.20, 25.62, 41.12, 47.77, 42.77]
y5 = [31.22, 26.76, 24.20, 35.81, 48.35, 23.35, 30.57, 25.42, 25.71, 41.69, 48.32, 42.91]

plt.plot(x, y1, '^:g', mfc="None", label='Layer1', linewidth=2.0)
plt.plot(x, y2, '*-.b', mfc="None", label='Layer2', linewidth=2.0)
plt.plot(x, y3, 'o--y', mfc="None", label='Layer3', linewidth=2.0)
plt.plot(x, y4, 'd-r', mfc="None", label='Layer4', linewidth=2.0)
plt.plot(x, y5, 'x:k', mfc="None", label='Layer5', linewidth=2.0)
plt.title("Number of heads of multi-head attention", fontproperties='Times New Roman', size=20)


plt.grid()
plt.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# plt.xticks(x,tick_label)
plt.legend(prop=font2)

plt.yticks(fontproperties='Times New Roman', size=18)
# plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.xticks(fontproperties='Times New Roman', size=18)

y_major_locator = MultipleLocator(20)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(10, 60)
# plt.ylim(20, 110)
# plt.ylim(10, 50)
# plt.ylim(0, 1)
plt.savefig('figure.svg', format='svg')
plt.show()
