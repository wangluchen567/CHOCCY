# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
实现性能指标收敛情况的可视化绘图
"""

import warnings
import numpy as np
from .animator import Frame
from .colormap import sample_colors
from ...types import MetricType
from ...core import VisualizationWarning
from typing import Optional, Dict, Union


def plot_metrics(history: Dict[str, Union[list, np.ndarray]],
                 n_iter: Optional[int] = None,
                 **kwargs) -> Optional[Frame]:
    """
    根据历史记录绘制性能指标收敛变化
    :param history: 性能指标收敛变化的历史记录字典
    :param n_iter: 当前迭代次数，显示在标题中
    :return: 绘图帧实例 (失败则为None)
    """
    # 从历史记录字典中读取指定的性能指标数据
    metrics_data = get_metrics_data(history, kwargs.get('metrics', 'first'))
    if not metrics_data:
        warnings.warn(
            "No metric data available for plotting. "
            "Possible causes: "
            "1) No metrics were configured for tracking (call `track_metrics()` first); "
            "2) The algorithm hasn't completed a run yet; "
            "3) The requested metrics are not recorded in history.",
            VisualizationWarning,
            stacklevel=2
        )
    # 初始化动画帧
    frame = Frame()
    # 获取颜色映射
    colors = sample_colors(len(metrics_data),
                           cmap_name=kwargs.get('color_map', 'tab10'))
    # 逐一绘制指标情况
    for (metric_name, values), color in zip(metrics_data.items(), colors):
        frame.add_line(np.arange(len(values)), values,
                       color=color,
                       alpha=kwargs.get('alpha', 1.0),
                       marker=kwargs.get('marker', '.'),
                       label=metric_name)
    frame.set_labels(xlabel='iteration')
    frame.set_ticklabel_format(axis='y', style='sci')
    # 设置对数y轴
    if kwargs.get('log_y', False):
        frame.set_yscale('log')
        frame.set_labels(ylabel='log(value)')
    else:
        frame.set_labels(ylabel='value')
    # 添加图例
    if len(metrics_data):
        frame.set_legend(loc=kwargs.get('legend_loc', 'upper right'))
    # 添加网格
    frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
    # 设置标题
    if n_iter is None:
        frame.set_title("Convergence of Performance Metrics")
    else:
        frame.set_title(f"Convergence of Performance Metrics (Iteration {n_iter})")
    return frame


def get_metrics_data(history_data: dict, metrics: Union[str, list, MetricType] = 'first'):
    """
    从历史记录字典中读取指定的性能指标数据
    :param history_data: 历史记录字典
    :param metrics: 读取指定的性能指标或使用位置指定(all:所有, first:第一个, end:最后一个)
    :return: 指定的性能指标
    """
    # 获取所有可用的指标键（保持插入顺序）
    available_keys = list(history_data.keys())

    if isinstance(metrics, str):
        if metrics == 'all':
            keys_to_fetch = available_keys
        elif metrics == 'first':
            keys_to_fetch = [available_keys[0]] if available_keys else []
        elif metrics == 'end':
            keys_to_fetch = [available_keys[-1]] if available_keys else []
        else:
            # 单个指标名称
            keys_to_fetch = [metrics]
    else:
        # 处理列表或MetricType枚举
        if isinstance(metrics, MetricType):
            keys_to_fetch = [metrics.value]
        else:
            # 假设是列表
            keys_to_fetch = []
            for item in metrics:
                if isinstance(item, MetricType):
                    keys_to_fetch.append(item.value)
                else:
                    keys_to_fetch.append(str(item))
    # 读取数据
    return {key: history_data[key] for key in keys_to_fetch if key in history_data}
