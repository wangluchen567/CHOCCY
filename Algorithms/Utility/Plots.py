import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def plot_scores(scores, score_type=None):
    """绘制评价指标情况"""
    plt.figure()
    plt.plot(np.arange(len(scores)), scores, marker=".", c="red")
    plt.grid()
    if score_type is not None:
        plt.title(score_type + " Scores")
    plt.show()


def plot_data(X, pause=False, n_iter=None, pause_time=0.1):
    """对任意维度数据进行绘图"""
    if not pause: plt.figure()
    plt.clf()
    X_dim = X.shape[1]
    if X_dim == 1:
        X_stack = np.hstack((X, X))
        for i in range(len(X)):
            plt.plot(np.arange(0, 2), X_stack[i, :])
        plt.xlim((0, 1))
    elif X_dim == 2:
        plt.scatter(X[:, 0], X[:, 1], marker="o", c="blue")
    elif X_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c="blue")
        # 设置三维图像角度(仰角方位角)
        # ax.view_init(elev=30, azim=30)
    else:
        for i in range(len(X)):
            plt.plot(np.arange(1, X_dim + 1), X[i, :])
    plt.grid()
    if pause:
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def plot_objs(objs, pause=False, n_iter=None, pause_time=0.1, pareto_front=None):
    """将最优目标值列表绘折线图"""
    # 先确定目标值维度
    if isinstance(objs, list):
        obj_dim = 1
    else:
        obj_dim = objs.shape[1]
    if not pause: plt.figure()
    plt.clf()
    if obj_dim == 1:
        plt.plot(np.arange(len(objs)), objs, marker=".", c="blue")
        plt.xlabel('N_iter')
        plt.ylabel('Fitness')
    elif obj_dim == 2:
        plt.scatter(objs[:, 0], objs[:, 1], marker="o", c="blue")
        # 绘制最优前沿面
        if pareto_front is not None:
            plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker="", c="red")
        plt.xlabel('F1')
        plt.ylabel('F2')
    elif obj_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], marker="o", c="blue")
        # 绘制最优前沿面
        if pareto_front is not None:
            x = pareto_front[:, 0]
            y = pareto_front[:, 1]
            z = pareto_front[:, 2]
            # 生成需要插值的网格
            xi = np.linspace(min(x), max(x), 100)
            yi = np.linspace(min(y), max(y), 100)
            xi, yi = np.meshgrid(xi, yi)
            # 使用 griddata 进行插值
            zi = griddata((x, y), z, (xi, yi), method='linear')
            ax.plot_wireframe(xi, yi, zi, color='red', linewidth=0.3)
        # 设置三维图像角度(仰角方位角)
        ax.view_init(elev=30, azim=30)
        ax.set_xlabel('F1')
        ax.set_ylabel('F2')
        ax.set_zlabel('F3')
    else:
        x = np.arange(1, obj_dim + 1)
        for i in range(len(objs)):
            plt.plot(x, objs[i, :])
    plt.grid()
    if pause:
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def plot_decs_objs(problem, decs, objs, pause=False, n_iter=None, pause_time=0.1, contour=True, sym=True):
    """绘制混合图像方便展示"""
    decs_dim = decs.shape[1]
    if objs.ndim == 1:
        objs = objs.reshape(-1, 1)
    objs_dim = objs.shape[1]
    if objs_dim > 1:
        raise ValueError("The target value dimension must be 1 dimension")
    if not pause: plt.figure()
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
        # x = np.linspace(problem.lower[0], problem.upper[0], 1000).reshape(-1, 1)
        # y = np.linspace(problem.lower[1], problem.upper[1], 1000).reshape(-1, 1)
        if sym is True:  # 对称图像绘制
            x_min, x_max = -np.max(np.abs(decs)), np.max(np.abs(decs))
            y_min, y_max = -np.max(np.abs(decs)), np.max(np.abs(decs))
        else:
            x_min, x_max = np.min(decs[:, 0]), np.max(decs[:, 0])
            y_min, y_max = np.min(decs[:, 1]), np.max(decs[:, 1])
        x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        y = np.linspace(y_min, y_max, 1000).reshape(-1, 1)
        X, Y = np.meshgrid(x, y)
        Z = problem.cal_objs(np.concatenate((np.expand_dims(X, -1), np.expand_dims(Y, -1)), -1))
        if contour:
            # 设置x轴和y轴使用科学计数法
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 20))
            # plt.clabel(contour, inline=True, fontsize=8)  # 添加等高线标签
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
    plt.grid()
    if pause:
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()
