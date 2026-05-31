# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
局部搜索算法集
Local Search
"""

from .LS_TSP import LocalSearch
from .GFLS_TSP import GuidedFastLocalSearch

# 提供别名
LocalSearchForTSP = LocalSearch


__all__ = [
    'LocalSearch',
    'LocalSearchForTSP',
    'GuidedFastLocalSearch'
]