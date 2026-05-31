# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
实现 决策空间/目标空间/性能指标 绘图的多个算法对比的可视化
"""

import numpy as np
from typing import Dict
from typing import Union
from typing import Optional
from .animator import Frame
from ...problems import Problem
from .colormap import sample_colors
from .decisions import plot_decisions
from .objectives import plot_objectives
from .objectives import plot_history_objs
from .hybrids import plot_hybrids_2d, plot_hybrids_3d


def plot_decision_comparison(decs_dict: Dict[str, np.ndarray],
                             n_iter: Optional[int] = None,
                             **kwargs) -> Optional[Frame]:
    """
    绘制多个算法的决策空间对比可视化（支持1D、2D、3D及高维数据）

    :param decs_dict: 决策变量字典，包含决策变量矩阵，形状为 (n_samples, n_vars)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 获取颜色映射
    colors = sample_colors(len(decs_dict),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 移除可能冲突的参数
    kwargs.pop('line_color', None)
    kwargs.pop('node_color', None)
    # 初始化动画帧列表
    frames = []
    for (name, decs), color in zip(decs_dict.items(), colors):
        # 添加当前算法动画帧
        frames.append(
            plot_decisions(
                decs, n_iter,
                line_color=color,
                node_color=color,
                label=name, **kwargs
            )
        )
    # 创建总的动画帧
    frame = Frame.merge_frames(frames)
    frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    return frame


def plot_objectives_comparison(objs_dict: Dict[str, np.ndarray],
                               n_iter: Optional[int] = None,
                               **kwargs) -> Optional[Frame]:
    """
    绘制多个算法的目标空间对比可视化（支持1D、2D、3D及高维数据）

    :param objs_dict: 目标值字典，目标值矩阵，形状为 (n_samples, n_objs)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 获取颜色映射
    colors = sample_colors(len(objs_dict),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 移除可能冲突的参数
    kwargs.pop('line_color', None)
    kwargs.pop('node_color', None)
    pareto_front = kwargs.pop('pareto_front', None)
    # 初始化动画帧列表
    frames = []
    items = list(objs_dict.items())
    # 绘制所有算法数据
    for i, ((name, objs), color) in enumerate(zip(items, colors)):
        # 只在最后一个算法显示Pareto前沿
        current_pareto = pareto_front \
            if i == len(items) - 1 else None
        # 添加当前算法动画帧
        frames.append(
            plot_objectives(
                objs, n_iter,
                line_color=color,
                node_color=color,
                label=name,
                pareto_front=current_pareto,
                **kwargs
            )
        )
    frame = Frame.merge_frames(frames)
    frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    return frame


def plot_history_comparison(history_dict: Dict[str, np.ndarray],
                            n_iter: Optional[int] = None,
                            **kwargs) -> Optional[Frame]:
    """
    绘制多个算法的历史记录目标值矩阵的变化的对比可视化

    :param history_dict: 目标值矩阵变化的历史记录字典
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 获取颜色映射
    colors = sample_colors(len(history_dict),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 移除可能冲突的参数
    kwargs.pop('line_color', None)
    kwargs.pop('fill_color', None)
    # 初始化动画帧列表
    frames = []
    for (name, decs), color in zip(history_dict.items(), colors):
        # 添加当前算法动画帧
        frames.append(
            plot_history_objs(
                decs, n_iter,
                line_color=color,
                fill_color=color,
                label=name, **kwargs
            )
        )
    # 创建总的动画帧
    frame = Frame.merge_frames(frames)
    frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    return frame


def plot_hybrids_2d_comparison(problem: Problem,
                               decs_dict: Dict[str, np.ndarray],
                               objs_dict: Dict[str, np.ndarray],
                               n_iter: Optional[int] = None,
                               **kwargs) -> Optional[Frame]:
    """
    实现多个算法的决策空间与目标空间混合绘制的对比可视化（绘制二维图像）（仅支持1D、2D决策向量的单目标问题）

    :param problem: 问题对象实例（必须拥有 calc_objs 函数）
    :param decs_dict: 决策变量字典，包含决策变量矩阵，形状为 (n_samples, n_vars)
    :param objs_dict: 目标值字典，目标值矩阵，形状为 (n_samples, 1)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 移除可能冲突的参数
    kwargs.pop('label', None)
    kwargs.pop('node_color', None)
    # 弹出绘图相关参数
    sym = kwargs.pop('sym', True)
    fixed = kwargs.pop('fixed', False)
    # 弹出范围相关参数
    kwargs.pop('show_samples', False)
    kwargs.pop('x_range', (None, None))
    x_range = kwargs.pop('x_range', (None, None))
    y_range = kwargs.pop('y_range', (None, None))
    all_decs = np.hstack(list(decs_dict.values()))
    if problem.n_vars == 1:
        if fixed:  # 固定视角绘图
            x_range = problem.l_bounds[0], problem.u_bounds[0]
        else:
            if sym:  # 绘制对称函数图像
                x_range = -np.max(np.abs(all_decs)), np.max(np.abs(all_decs))
            else:
                x_range = np.min(all_decs), np.max(all_decs)
    elif problem.n_vars == 2:
        if fixed:  # 固定视角绘图
            x_range = problem.l_bounds[0], problem.u_bounds[0]
            y_range = problem.l_bounds[1], problem.u_bounds[1]
        else:
            # 对问题进行采样绘制问题函数图像
            if sym:  # 完全对称函数绘图
                x_range = -np.max(np.abs(all_decs)), np.max(np.abs(all_decs))
                y_range = -np.max(np.abs(all_decs)), np.max(np.abs(all_decs))
            else:  # 非对称函数绘图
                x_range = np.min(all_decs[:, 0]), np.max(all_decs[:, 0])
                y_range = np.min(all_decs[:, 1]), np.max(all_decs[:, 1])

    # 获取颜色映射
    colors = sample_colors(len(decs_dict),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 初始化动画帧列表
    frames = []
    # 绘制所有算法数据
    for i, (name, color) in enumerate(zip(decs_dict.keys(), colors)):
        # 只在第一个算法显示问题采样图像
        show_samples = True if i == 0 else False
        # 添加当前算法动画帧
        frames.append(
            plot_hybrids_2d(
                problem,
                decs_dict[name],
                objs_dict[name],
                n_iter=n_iter,
                node_color=color,
                label=name,
                x_range=x_range,
                y_range=y_range,
                show_samples=show_samples,
                **kwargs
            )
        )
    # 创建总的动画帧
    frame = Frame.merge_frames(frames)
    frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    return frame


def plot_hybrids_3d_comparison(problem: Problem,
                               decs_dict: Dict[str, np.ndarray],
                               objs_dict: Dict[str, np.ndarray],
                               n_iter: Optional[int] = None,
                               **kwargs) -> Optional[Frame]:
    """
    实现多个算法的决策空间与目标空间混合绘制的对比可视化（绘制三维图像）（仅支持2D决策向量的单目标问题）

    :param problem: 问题对象实例（必须拥有 calc_objs 函数）
    :param decs_dict: 决策变量字典，包含决策变量矩阵，形状为 (n_samples, n_vars)
    :param objs_dict: 目标值字典，目标值矩阵，形状为 (n_samples, 1)
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 移除可能冲突的参数
    kwargs.pop('label', None)
    kwargs.pop('node_color', None)
    # 弹出绘图相关参数
    sym = kwargs.pop('sym', True)
    fixed = kwargs.pop('fixed', False)
    # 弹出范围相关参数
    kwargs.pop('show_samples', False)
    kwargs.pop('x_range', (None, None))
    x_range = kwargs.pop('x_range', (None, None))
    y_range = kwargs.pop('y_range', (None, None))
    all_decs = np.hstack(list(decs_dict.values()))
    if problem.n_vars == 2:
        if fixed:  # 固定视角绘图
            x_range = problem.l_bounds[0], problem.u_bounds[0]
            y_range = problem.l_bounds[1], problem.u_bounds[1]
        else:
            # 对问题进行采样绘制问题函数图像
            if sym:  # 完全对称函数绘图
                x_range = -np.max(np.abs(all_decs)), np.max(np.abs(all_decs))
                y_range = -np.max(np.abs(all_decs)), np.max(np.abs(all_decs))
            else:  # 非对称函数绘图
                x_range = np.min(all_decs[:, 0]), np.max(all_decs[:, 0])
                y_range = np.min(all_decs[:, 1]), np.max(all_decs[:, 1])

    # 获取颜色映射
    colors = sample_colors(len(decs_dict),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 初始化动画帧列表
    frames = []
    # 绘制所有算法数据
    for i, (name, color) in enumerate(zip(decs_dict.keys(), colors)):
        # 只在第一个算法显示问题采样图像
        show_samples = True if i == 0 else False
        # 添加当前算法动画帧
        frames.append(
            plot_hybrids_3d(
                problem,
                decs_dict[name],
                objs_dict[name],
                n_iter=n_iter,
                node_color=color,
                label=name,
                x_range=x_range,
                y_range=y_range,
                show_samples=show_samples,
                **kwargs
            )
        )
    # 创建总的动画帧
    frame = Frame.merge_frames(frames)
    frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    return frame


def plot_metrics_comparison(metric_name: str,
                            history: Dict[str, Union[list, np.ndarray]],
                            n_iter: Optional[int] = None,
                            **kwargs):
    """
    根据历史记录绘制性能指标收敛变化，实现多个算法对比可视化

    :param metric_name: 追踪的性能指标名称
    :param history: 性能指标收敛变化的历史记录字典
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 初始化动画帧
    frame = Frame()
    # 获取颜色映射
    colors = sample_colors(len(history),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 逐一绘制指标情况
    for (name, values), color in zip(history.items(), colors):
        frame.add_line(np.arange(len(values)), values,
                       color=color,
                       alpha=kwargs.get('alpha', 1.0),
                       marker=kwargs.get('marker', '.'),
                       label=name)
    frame.set_labels(xlabel='iteration')
    frame.set_ticklabel_format(axis='y', style='sci')
    # 设置对数y轴
    if kwargs.get('log_y', False):
        frame.set_yscale('log')
        frame.set_labels(ylabel='log(value)')
    else:
        frame.set_labels(ylabel='value')
    # 添加图例
    if len(history):
        frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title(f"Convergence of {metric_name.upper()} Metrics")
    else:
        frame.set_title(f"Convergence of {metric_name.upper()} Metrics (Iteration {n_iter})")
    return frame
