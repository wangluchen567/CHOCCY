# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
实现决策空间与目标空间的混合绘图
"""

import numpy as np
from typing import Optional
from .animator import Frame
from ...problems import Problem
from ...core import VisualizationWarning, warn_once


def plot_hybrids_2d(problem: Problem,
                    decs: np.ndarray,
                    objs: Optional[np.ndarray] = None,
                    n_iter: Optional[int] = None,
                    **kwargs) -> Optional[Frame]:
    """
    实现决策空间与目标空间混合绘制（绘制二维图像）（仅支持1D、2D决策向量的单目标问题）

    :param problem: 问题对象实例（必须拥有 calc_objs 函数）
    :param decs: 决策向量矩阵，形状为 (n_samples, n_vars)
    :param objs: 目标值矩阵，形状为 (n_samples, 1)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 得到变量个数(向量维度)
    n_vars = decs.shape[1]
    # 得到目标值矩阵
    objs = problem.calc_objs(decs) if objs is None else objs
    # 判断形状是否匹配
    if objs.shape[0] != decs.shape[0]:
        warn_once(f"Cannot visualize: objective rows ({objs.shape[0]}) != decision rows ({decs.shape[0]}).",
                  VisualizationWarning)
    # 判断是否目标数过多（仅支持单目标）
    if objs.shape[1] > 1:
        warn_once(f"Cannot visualize: multi-objective not supported (got {objs.shape[1]} objectives, need 1).",
                  VisualizationWarning)

    # 从 kwargs 中提取参数并设置默认值
    sym = kwargs.get('sym', True)
    fixed = kwargs.get('fixed', False)
    # 问题采样设置
    n_samples = kwargs.get('n_samples', 1000)
    show_samples = kwargs.get('show_samples', True)
    # 等高线配置
    fill_contour = kwargs.get('fill_contour', True)
    add_contour = kwargs.get('add_contour', False)
    num_contours = kwargs.get('num_contours', 16)
    level_range = kwargs.get('level_range', (None, None))
    add_clabel = kwargs.get('add_clabel', False)
    # 颜色相关配置
    contourf_color_map = kwargs.get('contourf_color_map', 'viridis')
    show_color_bar = kwargs.get('show_color_bar', False)
    # 问题采样绘制图像的范围
    x_range = kwargs.get('x_range', (None, None))
    y_range = kwargs.get('y_range', (None, None))

    # 根据决策向量维度选择绘图方式
    if n_vars == 1:
        # 初始化动画帧
        frame = Frame()
        # 1D决策空间 + 1D目标空间
        # 检查绘制问题采样的设置
        if show_samples:
            # 对问题进行采样绘制问题函数图像
            if not isinstance(x_range, tuple) or None in x_range:
                if fixed:  # 固定视角绘图
                    x_range = problem.l_bounds[0], problem.u_bounds[0]
                else:
                    if sym:  # 绘制对称函数图像
                        x_range = -np.max(np.abs(decs)), np.max(np.abs(decs))
                    else:
                        x_range = np.min(decs), np.max(decs)
            # 绘制问题的函数图像
            x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
            y = problem.calc_objs(x)
            frame.add_line(x, y, color=kwargs.get('problem_color', 'blue'))

            # 绘制给定决策变量矩阵与目标值矩阵情况
            frame.add_scatter(decs, objs,
                              marker=kwargs.get('marker', 'o'),  # 默认是圆形
                              c=kwargs.get('node_color', 'red'),  # 默认颜色为红色
                              alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                              s=kwargs.get('size', 30),  # 默认大小为 30
                              edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                              linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                              label=kwargs.get('label', None))  # 默认标签为空
            frame.set_labels(xlabel='x', ylabel='obj')
            frame.set_ticklabel_format(axis='both', style='sci')
    elif n_vars == 2:
        # 初始化动画帧
        frame = Frame()
        # 2D决策空间 + 1D目标空间
        # 检查绘制问题采样的设置
        if show_samples:
            if (not isinstance(x_range, tuple) or None in x_range or
                    not isinstance(y_range, tuple) or None in y_range):
                if fixed:  # 固定视角绘图
                    x_range = problem.l_bounds[0], problem.u_bounds[0]
                    y_range = problem.l_bounds[1], problem.u_bounds[1]
                else:
                    # 对问题进行采样绘制问题函数图像
                    if sym:  # 完全对称函数绘图
                        x_range = -np.max(np.abs(decs)), np.max(np.abs(decs))
                        y_range = -np.max(np.abs(decs)), np.max(np.abs(decs))
                    else:  # 非对称函数绘图
                        x_range = np.min(decs[:, 0]), np.max(decs[:, 0])
                        y_range = np.min(decs[:, 1]), np.max(decs[:, 1])
            x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
            y = np.linspace(y_range[0], y_range[1], n_samples).reshape(-1, 1)
            xs, ys = np.meshgrid(x, y)
            # 将输入的决策变量打包
            decs_pack = np.concatenate((np.expand_dims(xs, -1), np.expand_dims(ys, -1)), -1).reshape(-1, 2)
            objs_pack = problem.calc_objs(decs_pack)  # 计算所有采样数据的目标值
            zs = objs_pack.reshape(xs.shape)  # 将形状转换为与xs匹配
            # 等高线的数值范围设置
            if not isinstance(level_range, tuple) or None in level_range:
                level_range = np.min(zs), np.max(zs)
            # 绘制问题函数图像（使用等高线方法绘制）
            if fill_contour:  # 若使用填充绘制（速度较慢）
                # 先绘制填充
                contourf_ = frame.add_contourf(xs, ys, zs,
                                               levels=np.linspace(level_range[0], level_range[1], num_contours),
                                               cmap=contourf_color_map, alpha=0.8)
                if add_contour:
                    # 绘制轮廓线
                    contour_ = frame.add_contour(xs, ys, zs,
                                                 levels=np.linspace(level_range[0], level_range[1], num_contours),
                                                 colors='black', linewidths=0.5)
                    # 添加等高线标签
                    if add_clabel:
                        frame.add_contour_labels(contour_, inline=True, fontsize=6)
                # 添加颜色条信息
                if show_color_bar:
                    frame.add_colorbar(contourf_)
            else:  # 若使用非填充绘制
                contour_ = frame.add_contour(xs, ys, zs,
                                             levels=np.linspace(level_range[0], level_range[1], num_contours))
                # 添加等高线标签
                if add_clabel:
                    frame.add_contour_labels(contour_, inline=True, fontsize=6)
        # 绘制给定决策变量矩阵情况
        frame.add_scatter(decs[:, 0], decs[:, 1],
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'red'),  # 默认颜色为红色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None))  # 默认标签为空
        frame.set_labels(xlabel='$x^{(1)}$', ylabel='$x^{(2)}$')
        frame.set_ticklabel_format(axis='both', style='sci')
    else:
        warn_once(f"Cannot visualize: decision vector dimension must be 1 or 2, but got {n_vars}.",
                  VisualizationWarning)
        return None
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Hybrids Space")
    else:
        frame.set_title(f"Hybrids Space (Iteration {n_iter})")
    return frame


def plot_hybrids_3d(problem: Problem,
                    decs: np.ndarray,
                    objs: Optional[np.ndarray] = None,
                    n_iter: Optional[int] = None,
                    **kwargs) -> Optional[Frame]:
    """
    实现决策空间与目标空间混合绘制（绘制三维图像）（仅支持2D决策向量的单目标问题）

    :param problem: 问题对象实例（必须拥有 calc_objs 函数）
    :param decs: 决策向量矩阵，形状为 (n_samples, n_vars)
    :param objs: 目标值矩阵，形状为 (n_samples, 1)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """

    # 得到变量个数(向量维度)
    n_vars = decs.shape[1]
    # 得到目标值矩阵
    objs = problem.calc_objs(decs) if objs is None else objs
    # 判断形状是否匹配
    if objs.shape[0] != decs.shape[0]:
        warn_once(f"Cannot visualize: objective rows ({objs.shape[0]}) != decision rows ({decs.shape[0]}).",
                  VisualizationWarning)
    # 判断是否目标数过多（仅支持单目标）
    if objs.shape[1] > 1:
        warn_once(f"Cannot visualize: multi-objective not supported (got {objs.shape[1]} objectives, need 1).",
                  VisualizationWarning)

    # 从 kwargs 中提取参数并设置默认值
    sym = kwargs.get('sym', True)
    fixed = kwargs.get('fixed', False)
    # 问题采样设置
    n_samples = kwargs.get('n_samples', 1000)
    show_samples = kwargs.get('show_samples', True)
    surface_color_map = kwargs.get('surface_color_map', 'viridis')
    show_color_bar = kwargs.get('show_color_bar', False)
    surface_alpha = kwargs.get('surface_alpha', 0.5)
    # 问题采样绘制图像的范围
    x_range = kwargs.get('x_range', (None, None))
    y_range = kwargs.get('y_range', (None, None))

    # 根据决策向量维度选择绘图方式
    if n_vars == 2:
        # 初始化动画帧(3D)
        frame = Frame(is_3d=True)
        # 2D决策空间 + 1D目标空间，使用三维空间方法绘制
        if show_samples:
            # 对问题进行采样绘制问题函数图像
            if (not isinstance(x_range, tuple) or None in x_range or
                    not isinstance(y_range, tuple) or None in y_range):
                if fixed:  # 固定视角绘图
                    x_range = problem.l_bounds[0], problem.u_bounds[0]
                    y_range = problem.l_bounds[1], problem.u_bounds[1]
                else:
                    if sym:  # 完全对称函数绘图
                        x_range = -np.max(np.abs(decs)), np.max(np.abs(decs))
                        y_range = -np.max(np.abs(decs)), np.max(np.abs(decs))
                    else:  # 非对称函数绘图
                        x_range = np.min(decs[:, 0]), np.max(decs[:, 0])
                        y_range = np.min(decs[:, 1]), np.max(decs[:, 1])
            x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
            y = np.linspace(y_range[0], y_range[1], n_samples).reshape(-1, 1)
            xs, ys = np.meshgrid(x, y)
            # 将输入的决策变量打包
            decs_pack = np.concatenate((np.expand_dims(xs, -1), np.expand_dims(ys, -1)), -1).reshape(-1, 2)
            objs_pack = problem.calc_objs(decs_pack)  # 计算所有采样数据的目标值
            zs = objs_pack.reshape(xs.shape)  # 将形状转换为与xs匹配
            # 绘制问题函数 三维面绘制
            surface_ = frame.add_surface(xs, ys, zs, cmap=surface_color_map, alpha=surface_alpha, zorder=1)
            if show_color_bar:
                # 添加颜色条信息
                frame.add_colorbar(surface_, shrink=0.6, aspect=20, pad=0.1)
        # 绘制给定决策变量矩阵与目标值矩阵情况
        frame.add_scatter(decs[:, 0], decs[:, 1], objs.flatten(),
                          marker=kwargs.get('marker', 'o'),  # 默认是圆形
                          c=kwargs.get('node_color', 'red'),  # 默认颜色为红色
                          alpha=kwargs.get('alpha', 1.0),  # 默认不透明
                          s=kwargs.get('size', 30),  # 默认大小为 30
                          edgecolor=kwargs.get('edge_color', 'black'),  # 默认边缘颜色为黑色
                          linewidth=kwargs.get('edge_width', 0.1),  # 默认边缘宽度为 0.1
                          label=kwargs.get('label', None),  # 默认标签为空
                          zorder=2)
        frame.set_labels(xlabel='$x^{(1)}$', ylabel='$x^{(2)}$', zlabel='obj')
        frame.set_ticklabel_format(axis='both', style='sci')
    else:
        warn_once(f"Cannot visualize: decision vector dimension must be 2, but got {n_vars}.",
                  VisualizationWarning)
        return None
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Hybrids Space")
    else:
        frame.set_title(f"Hybrids Space (Iteration {n_iter})")
    return frame
