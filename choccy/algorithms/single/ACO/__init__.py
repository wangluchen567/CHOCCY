# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
蚁群算法集
Ant Colony Algorithms
"""

from .ACO import ACO

# 提供别名
AntColony = ACO

__all__ = [
    'ACO',
    'AntColony'
]