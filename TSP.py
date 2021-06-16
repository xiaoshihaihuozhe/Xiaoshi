#!/usr/bin/python
# _*_ coding:utf-8 _*_
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from numpy.matlib import rand
#from matplotlib.mlab import dist
from matplotlib.artist import getp
import copy

'''
记录错误，数组直接复制是复制地址
例如， current = route
想要得到一个新的有同样内容的数组，应该用： current = copy.copy(route) 
'''

# 初始三十个城市坐标
city_x = [6, 12, 19, 52, 84, 87, 71, 71,58, 80,46,43,77,90,23,1,4,66,22,87]
city_y = [41, 42, 38, 57, 59, 62, 65, 77, 68, 66,22,1,55,67,88,43,22,66,54,79]
# 城市数量
city_num = 20
plt.plot(city_x,city_y,'o')
distance = [[0 for col in range(city_num)] for raw in range(city_num)] #列表推导式 生成一个20*20的0矩阵，用来存储距离，第一维长度为你，第二维长度为n
# 初始温度 结束温度
T0 = 30
Tend = 1e-8
# 循环控制常数
L = 30
# 温度衰减系数
a = 0.98


# 构建初始参考距离矩阵init_
def init_distance():
    for i in range(city_num):
        for j in range(city_num):
            x = pow(city_x[i] - city_x[j], 2)  #pow(x,y) = x的y次幂
            y = pow(city_y[i] - city_y[j], 2)
            distance[i][j] = pow(x + y, 0.5)  #distance存放x点到y点的距离，计算20个点两两之间的距离
    for i in range(city_num):
        for j in range(city_num):
            if distance[i][j] == 0:
                distance[i][j] = sys.maxsize #取init的最大值


# 计算总距离
def route_total_dist(rou):
    sum = 0.0
    for i in range(city_num - 1):
        sum += distance[rou[i]][rou[i + 1]]
    sum += distance[rou[city_num - 1]][rou[0]]  #最后加上回归的初始点的距离
    return sum


# 得到新解
def getnewroute(route, time):   #输入路径，当前迭代次数
    # 如果是偶数次，二变换法
    current = copy.copy(route) #复制route的内容，并且存放到另外一个地址

    if time % 2 == 0:
        u = random.randint(0, city_num - 1)  #随机生成一个 0到city——num-1的整数
        v = random.randint(0, city_num - 1)
        temp = current[u]
        current[u] = current[v]
        current[v] = temp
    # 如果是奇数次，三变换法
    else:
        temp2 = random.sample(range(0, city_num), 3)  #从0到city_num中提取3个数
        temp2.sort()  #从小到大排序
        u = temp2[0]
        v = temp2[1]
        w = temp2[2]
        w1 = w + 1
        temp3 = [0 for col in range(v - u + 1)]
        j = 0
        for i in range(u, v + 1):
            temp3[j] = current[i]
            j += 1

        for i2 in range(v + 1, w + 1):
            current[i2 - (v - u + 1)] = current[i2]
        w = w - (v - u + 1)
        j = 0
        for i3 in range(w + 1, w1):
            current[i3] = temp3[j]
            j += 1

    return current


def draw(best):
    result_x = [0 for col in range(city_num + 1)]
    result_y = [0 for col in range(city_num + 1)]

    for i in range(city_num):
        result_x[i] = city_x[best[i]]
        result_y[i] = city_y[best[i]]
    result_x[city_num] = result_x[0]
    result_y[city_num] = result_y[0]
    print(result_x)
    print(result_y)
    plt.xlim(0, 100)  # 限定横轴的范围
    plt.ylim(0, 100)  # 限定纵轴的范围
    plt.plot(result_x, result_y, marker='>', mec='r', mfc='w', label=u'Route')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"x")  # X轴标签
    plt.ylabel(u"y")  # Y轴标签
    plt.title("TSP Solution")  # 标题

    plt.show()
    plt.close(0)


def solve():
    # 得到距离矩阵
    init_distance()
    # 得到初始解以及初始距离
    route = random.sample(range(0, city_num), city_num) #从rang(0, city_num)中随机提取city_num个数字，相当于打乱20个数字，随机生成一条路线
    total_dist = route_total_dist(route)  #计算得到路径的总距离
    print("初始路线：", route)
    print("初始距离：", total_dist)
    # 新解
    newroute = []
    new_total_dist = 0.0
    best = route
    best_total_dist = total_dist
    t = T0  #初始温度

    while True:
        if t <= Tend:  #如果当前温度小于结束温度，则跳出循环
            break
        # 令温度为初始温度
        for rt2 in range(L):  #循环控制常数 L=10
            newroute = getnewroute(route, rt2)
            new_total_dist = route_total_dist(newroute)
            delt = new_total_dist - total_dist
            if delt <= 0:
                route = newroute
                total_dist = new_total_dist
                if best_total_dist > new_total_dist:
                    best = newroute
                    best_total_dist = new_total_dist
            elif delt > 0:
                p = math.exp(-delt / t)
                ranp = random.uniform(0, 1)
                if ranp < p:
                    route = newroute
                    total_dist = new_total_dist
        t = t * a
    print("现在温度为：", t)
    print("最佳路线：", best)
    print("最佳距离：", best_total_dist)
    draw(best)


if __name__ == "__main__":
    solve()

