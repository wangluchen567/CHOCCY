# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
梯度下降算法集
Gradient Descent
"""

from .Adam import Adam
from .GD import GradientDecent

__all__ = [
    'Adam',
    'GradientDecent',
]