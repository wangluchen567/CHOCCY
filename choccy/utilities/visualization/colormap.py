# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
颜色相关函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def sample_colors(num_colors: int,
                  cmap_name: str = 'rainbow') -> list:
    """
    从颜色图谱(colormap)中采样颜色
    :param num_colors: 采样颜色的数量
    :param cmap_name: 采样图谱(colormap)的名称（默认为rainbow）
    :return: 颜色列表（十六进制表示）
    """
    # 数量少时返回常用颜色
    if num_colors <= 3:
        raw_colors = ['red', 'blue', 'green'][:num_colors]
    else:
        # 获取colormap
        try:
            cmap = plt.colormaps[cmap_name]
        except (AttributeError, KeyError):
            cmap = plt.cm.get_cmap(cmap_name)
        # 生成并转换颜色
        raw_colors = cmap(np.linspace(1, 0, num_colors))
    colors = [mcolors.to_hex(c) for c in raw_colors]
    return colors
