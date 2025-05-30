"""
绘图工具
Plot Utils

Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_decs(X, n_iter=None, pause=False, pause_time=0.06):
    """
    对任意维度决策向量数据进行绘图
    :param X: 决策向量组成的矩阵
    :param n_iter: 当前迭代次数
    :param pause: 是否短暂暂停显示
    :param pause_time: 短暂暂停的时间
    """
    plt.clf()
    X_dim = X.shape[1]
    if X_dim == 1:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        X_stack = np.hstack((X, X))
        for i in range(len(X)):
            plt.plot(np.arange(0, 2), X_stack[i, :])
        plt.xlim((0, 1))
        plt.xlabel('dim')
        plt.ylabel('x')
    elif X_dim == 2:
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.scatter(X[:, 0], X[:, 1], marker="o", c="blue")
        plt.xlabel('x')
        plt.ylabel('y')
    elif X_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c="blue")
        # 设置三维图像角度(仰角方位角)
        # ax.view_init(elev=30, azim=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        for i in range(len(X)):
            plt.plot(np.arange(1, X_dim + 1), X[i, :])
        plt.xlabel('dim')
        plt.ylabel('x')
    plt.grid(True)
    if n_iter is None:
        plt.title("Decisions")
    else:
        plt.title("iter: " + str(n_iter))
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()


def plot_objs(objs, n_iter=None, pause=False, pause_time=0.06, pareto_front=None):
    """
    对任意维度目标向量数据进行绘图
    :param objs: 目标向量组成的矩阵
    :param n_iter: 当前迭代次数
    :param pause: 是否短暂暂停显示
    :param pause_time: 短暂暂停的时间
    :param pareto_front: pareto最优前沿
    """
    # 先确定目标值维度
    if isinstance(objs, list):
        obj_dim = 1
    else:
        obj_dim = objs.shape[1]
    plt.clf()
    if obj_dim == 1:
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        x = np.arange(len(objs))
        objs_ = np.concatenate(objs, axis=1).T
        objs_min = np.min(objs_, axis=1)
        objs_max = np.max(objs_, axis=1)
        # 填充最小值和最大值之间的区域
        plt.fill_between(x, objs_min, objs_max, color='blue', alpha=0.2)
        plt.plot(x, objs_min, marker=".", c="blue")
        plt.xlabel('n_iter')
        plt.ylabel('fitness')
    elif obj_dim == 2:
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.scatter(objs[:, 0], objs[:, 1], marker="o", c="blue")
        # 绘制最优前沿面
        if pareto_front is not None:
            plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker="", c="gray")
        plt.xlabel('obj1')
        plt.ylabel('obj2')
    elif obj_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], marker="o", c="blue")
        # 绘制最优前沿面
        if pareto_front is not None:
            x = pareto_front[:, 0]
            y = pareto_front[:, 1]
            z = pareto_front[:, 2]
            # 绘制三维三角曲面（面透明且边浅灰）（支持不规则图形）
            ax.plot_trisurf(x, y, z, edgecolor='gray', color=(1, 1, 1, 0), linewidth=0.16)
        # 设置三维图像角度(仰角方位角)
        ax.view_init(elev=30, azim=30)
        ax.set_xlabel('obj1')
        ax.set_ylabel('obj2')
        ax.set_zlabel('obj3')
    else:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        x = np.arange(1, obj_dim + 1)
        for i in range(len(objs)):
            plt.plot(x, objs[i, :])
        plt.xlabel('dim')
        plt.ylabel('obj')
    plt.grid(True)
    if n_iter is None:
        plt.title("Objectives")
    else:
        plt.title("iter: " + str(n_iter))
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()


def plot_objs_decs(problem, decs, objs, n_iter=None, pause=False, pause_time=0.06, contour=True, sym=True):
    """
    在特定条件下可将目标空间与决策空间绘制到同一空间中
    :param problem: 问题对象
    :param decs: 决策向量（必须小于3维）组成的矩阵
    :param objs: 目标向量（必须等于1维）组成的矩阵
    :param n_iter: 当前迭代次数
    :param pause: 是否短暂暂停显示
    :param pause_time: 短暂暂停的时间
    :param contour: 是否使用等高线的方式绘制
    :param sym: 是否进行完全对称图像的绘制
    """
    decs_dim = decs.shape[1]
    if objs.ndim == 1:
        objs = objs.reshape(-1, 1)
    objs_dim = objs.shape[1]
    if objs_dim > 1:
        raise ValueError("The objective vector dimension must be 1 dimension")
    plt.clf()
    if decs_dim == 1:
        plt.scatter(decs, objs, marker="o", c="red")
        # 对问题进行采样绘制问题图像
        if sym is True:  # 对称图像绘制
            x_min, x_max = -np.max(np.abs(decs)), np.max(np.abs(decs))
        else:
            x_min, x_max = np.min(decs), np.max(decs)
        x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        y = problem.cal_objs(x)
        # 设置x轴和y轴使用科学计数法
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('obj')
    elif decs_dim == 2:
        # 对问题进行采样绘制问题图像
        if sym is True:  # 完全对称绘图
            x_min, x_max = -np.max(np.abs(decs)), np.max(np.abs(decs))
            y_min, y_max = -np.max(np.abs(decs)), np.max(np.abs(decs))
        else:  # 非对称绘图
            x_min, x_max = np.min(decs[:, 0]), np.max(decs[:, 0])
            y_min, y_max = np.min(decs[:, 1]), np.max(decs[:, 1])
        x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        y = np.linspace(y_min, y_max, 1000).reshape(-1, 1)
        X, Y = np.meshgrid(x, y)
        # 将输入的决策变量打包
        decs_pack = np.concatenate((np.expand_dims(X, -1), np.expand_dims(Y, -1)), -1).reshape(-1, 2)
        objs_pack = problem.cal_objs(decs_pack)
        Z = objs_pack.reshape(X.shape)
        if contour:
            # 设置x轴和y轴使用科学计数法
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            contour_ = plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 20))
            plt.clabel(contour_, inline=True, fontsize=8)  # 添加等高线标签
            plt.scatter(decs[:, 0], decs[:, 1], marker="o", c="red")
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            ax = plt.subplot(111, projection='3d')
            # 设置x轴、y轴和z轴使用科学计数法
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=1)
            ax.scatter(decs[:, 0], decs[:, 1], objs.flatten(), marker="o", s=50, c="red", zorder=2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('obj')
    else:
        raise ValueError("The decision vector dimension must be less than 3 dimensions")
    plt.grid(True)
    if n_iter is None:
        plt.title("Objs-Decs")
    else:
        plt.title("iter: " + str(n_iter))
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()


def plot_scores(scores, score_type=None, n_iter=None, pause=False, pause_time=0.06):
    """
    绘制评价指标变化情况
    :param scores: 评价指标数组
    :param score_type: 评价指标类型
    :param n_iter: 当前迭代次数
    :param pause: 是否短暂暂停显示
    :param pause_time: 短暂暂停的时间
    """
    plt.clf()
    plt.plot(np.arange(len(scores)), scores, marker=".", c="red")
    plt.xlabel('n_iter')
    if score_type is not None:
        plt.ylabel(score_type)
    plt.grid(True)
    if n_iter is None:
        if score_type is not None:
            plt.title(score_type + " Scores")
    else:
        plt.title("iter: " + str(n_iter))
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()
