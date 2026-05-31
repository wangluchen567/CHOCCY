# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
实现决策空间的绘图可视化
"""

import numpy as np
from typing import Optional
from .animator import Frame


def plot_decisions(decs: np.ndarray,
                   n_iter: Optional[int] = None,
                   **kwargs) -> Optional[Frame]:
    """
    绘制决策空间实现可视化（支持1D、2D、3D及高维数据）
    :param decs: 决策变量矩阵，形状为 (n_samples, n_vars)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 获取变量个数(向量维度)
    n_vars = decs.shape[1]
    # 根据向量维度选择绘图方式
    if n_vars == 1:
        # 初始化动画帧
        frame = Frame()
        # 1D决策空间：多线图
        decs_stack = np.hstack((decs, decs))
        # 获取多线的颜色
        line_color = kwargs.get('line_color', None)
        if line_color:  # 若指定多线的颜色则按照颜色绘制
            for i in range(len(decs)):
                frame.add_line(np.arange(0, 2), decs_stack[i, :],
                               c=line_color, label=kwargs.get('label', None))
        else:  # 若未指定多线颜色则按照T10调色板循环绘制
            for i in range(len(decs)):
                frame.add_line(np.arange(0, 2), decs_stack[i, :],
                               label=kwargs.get('label', None))
        frame.set_limits(xlim=(0, 1))
        frame.set_labels(xlabel='dim', ylabel='x')
        frame.set_ticklabel_format(axis='y', style='sci')
    elif n_vars == 2:
        # 初始化动画帧
        frame = Frame()
        # 2D决策空间：散点图
        frame.add_scatter(decs[:, 0], decs[:, 1],
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'blue'),  # 默认颜色为蓝色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None))  # 默认标签为空
        frame.set_labels(xlabel='$x^{(1)}$', ylabel='$x^{(2)}$')
        frame.set_ticklabel_format(axis='both', style='sci')
    elif n_vars == 3:
        # 初始化动画帧(3D)
        frame = Frame(is_3d=True)
        # 3D决策空间：散点图
        frame.add_scatter(decs[:, 0], decs[:, 1], decs[:, 2],
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'blue'),  # 默认颜色为蓝色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None))  # 默认标签为空
        # 设置三维图像角度(仰角方位角)
        frame.set_view(elev=kwargs.get('elev', 30), azim=kwargs.get('azim', 30))
        frame.set_labels(xlabel='$x^{(1)}$', ylabel='$x^{(2)}$', zlabel='$x^{(3)}$')
        frame.set_ticklabel_format(axis='both', style='sci')
    else:
        # 初始化动画帧
        frame = Frame()
        # 获取多线的颜色
        line_color = kwargs.get('line_color', None)
        if line_color:  # 若指定多线的颜色则按照颜色绘制
            for i in range(len(decs)):
                # 高维决策空间：平行坐标图
                frame.add_line(np.arange(1, n_vars + 1), decs[i, :],
                               c=line_color, label=kwargs.get('label', None))
        else:  # 若未指定多线颜色则按照T10调色板循环绘制
            for i in range(len(decs)):
                # 高维决策空间：平行坐标图
                frame.add_line(np.arange(1, n_vars + 1), decs[i, :],
                               label=kwargs.get('label', None))
        frame.set_labels(xlabel='dim', ylabel='x')
        frame.set_ticklabel_format(axis='y', style='sci')
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Decision Space")
    else:
        frame.set_title(f"Decision Space (Iteration {n_iter})")
    return frame
