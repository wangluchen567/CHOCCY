# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
差分进化算法集
Differential Evolution Algorithms
"""

from .DE import DE

# 提供别名
DifferentialEvolution = DE

__all__ = ['DE', 'DifferentialEvolution']