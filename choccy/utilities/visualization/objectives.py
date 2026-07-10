# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
实现目标空间的绘图可视化
"""

import numpy as np
from typing import Optional
from .animator import Frame


def plot_objectives(objs: np.ndarray,
                    n_iter: Optional[int] = None,
                    **kwargs) -> Optional[Frame]:
    """
    绘制目标空间实现可视化（支持1D、2D、3D及高维数据）
    :param objs: 目标值矩阵，形状为 (n_samples, n_objs)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 获取目标个数
    n_objs = objs.shape[1]
    # 根据目标个数选择绘图方式
    if n_objs == 1:
        # 初始化动画帧
        frame = Frame()
        # 1D目标空间：多线图
        objs_stack = np.hstack((objs, objs))
        # 获取多线的颜色
        line_color = kwargs.get('line_color', None)
        if line_color:  # 若指定多线的颜色则按照颜色绘制
            for i in range(len(objs)):
                frame.add_line(np.arange(0, 2), objs_stack[i, :],
                               c=line_color, label=kwargs.get('label', None))
        else:  # 若未指定多线颜色则按照T10调色板循环绘制
            for i in range(len(objs)):
                frame.add_line(np.arange(0, 2), objs_stack[i, :],
                               label=kwargs.get('label', None))
        frame.set_limits(xlim=(0, 1))
        frame.set_labels(xlabel='dim', ylabel='obj')
        frame.set_ticklabel_format(axis='y', style='sci')
    elif n_objs == 2:
        # 初始化动画帧
        frame = Frame()
        # 2D目标空间：散点图
        frame.add_scatter(objs[:, 0], objs[:, 1],
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'blue'),  # 默认颜色为蓝色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None))  # 默认标签为空
        # 获取帕累托最优前沿数据
        pareto_front = kwargs.get('pareto_front', None)
        # 获取最优解的点数据
        optimums = kwargs.get('optimums', None)
        # 绘制最优前沿面
        if pareto_front is not None:
            frame.add_line(pareto_front[:, 0], pareto_front[:, 1], marker="", c="gray")
        elif optimums is not None:
            frame.add_scatter(optimums[:, 0], optimums[:, 1], c="gray", s=10)
        frame.set_labels(xlabel='obj 1', ylabel='obj 2')
        frame.set_ticklabel_format(axis='both', style='sci')
    elif n_objs == 3:
        # 初始化动画帧(3D)
        frame = Frame(is_3d=True)
        # 3D目标空间：散点图
        frame.add_scatter(objs[:, 0], objs[:, 1], objs[:, 2],
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'blue'),  # 默认颜色为蓝色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None))  # 默认标签为空
        # 获取帕累托最优前沿数据
        pareto_front = kwargs.get('pareto_front', None)
        # 获取最优解的点数据
        optimums = kwargs.get('optimums', None)
        # 绘制最优前沿面
        if pareto_front is not None:
            if isinstance(pareto_front, list) and len(pareto_front) >= 3:  # 使用规则矩形网格绘制
                frame.add_wireframe(pareto_front[0], pareto_front[1], pareto_front[2],
                                    rstride=1, cstride=1, color='silver', linewidth=0.8)
            elif isinstance(pareto_front, np.ndarray) and pareto_front.shape[1] >= 3:  # 使用三角曲面绘制（适合不规则形状）
                frame.add_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                                  edgecolor='gray', color=(1, 1, 1, 0), linewidth=0.16)
        elif optimums is not None:
            frame.add_scatter(optimums[:, 0], optimums[:, 1], optimums[:, 2], c="gray", s=10)
        # 设置三维图像角度(仰角方位角)
        frame.set_view(elev=kwargs.get('elev', 30), azim=kwargs.get('azim', 30))
        frame.set_labels(xlabel='obj 1', ylabel='obj 2', zlabel='obj 3')
        frame.set_ticklabel_format(axis='both', style='sci')
    else:
        # 初始化动画帧
        frame = Frame()
        # 获取多线的颜色
        line_color = kwargs.get('line_color', None)
        if line_color:  # 若指定多线的颜色则按照颜色绘制
            for i in range(len(objs)):
                # 高维目标空间：平行坐标图
                frame.add_line(np.arange(1, n_objs + 1), objs[i, :], c=line_color)
        else:  # 若未指定多线颜色则按照T10调色板循环绘制
            for i in range(len(objs)):
                # 高维目标空间：平行坐标图
                frame.add_line(np.arange(1, n_objs + 1), objs[i, :])
        frame.set_labels(xlabel='dim', ylabel='obj')
        frame.set_ticklabel_format(axis='y', style='sci')
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Objective Space")
    else:
        frame.set_title(f"Objective Space (Iteration {n_iter})")
    return frame


def plot_history_objs(history: list[np.ndarray],
                      n_iter: Optional[int] = None,
                      **kwargs) -> Optional[Frame]:
    """
    根据历史记录绘制目标值矩阵的变化
    :param history: 目标值矩阵变化的历史记录
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 初始化动画帧
    frame = Frame()
    x = np.arange(len(history))
    objs_mat = np.concatenate(history, axis=1).T
    objs_min = np.min(objs_mat, axis=1)
    objs_max = np.max(objs_mat, axis=1)
    # 填充最小值和最大值之间的区域
    frame.add_fill_between(x, objs_min, objs_max,
                           color=kwargs.get('fill_color', 'blue'),
                           alpha=kwargs.get('fill_alpha', 0.2))
    frame.add_line(x, objs_min,
                   marker=kwargs.get('marker', '.'),  # 默认形状是点
                   c=kwargs.get('line_color', 'blue'),  # 默认颜色为蓝色
                   alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                   label=kwargs.get('label', None))  # 默认标签为空
    frame.set_labels(xlabel='iteration')
    frame.set_ticklabel_format(axis='y', style='sci')
    # 设置对数y轴
    if kwargs.get('log_y', False):
        frame.set_yscale('log')
        frame.set_labels(ylabel='log(obj)')
    else:
        frame.set_labels(ylabel='obj')
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Convergence of Objective Value")
    else:
        frame.set_title(f"Convergence of Objective Value (Iteration {n_iter})")
    return frame
