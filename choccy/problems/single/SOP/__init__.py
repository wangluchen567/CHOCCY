# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
单目标优化 基准测试问题 - SOP系列问题
"""

from .Ackley import Ackley
from .Sphere import Sphere
from .SOP1 import SOP1
from .SOP2 import SOP2
from .SOP3 import SOP3
from .SOP4 import SOP4
from .SOP5 import SOP5
from .SOP6 import SOP6
from .SOP7 import SOP7
from .SOP8 import SOP8
from .SOP9 import SOP9
from .SOP10 import SOP10

# 提供别名
Rastrigin = SOP9

__all__ = [
    'Ackley',
    'Sphere',
    'Rastrigin',
    'SOP1',
    'SOP2',
    'SOP3',
    'SOP4',
    'SOP5',
    'SOP6',
    'SOP7',
    'SOP8',
    'SOP9',
    'SOP10',
]