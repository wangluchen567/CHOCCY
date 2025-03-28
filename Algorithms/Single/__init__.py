"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
# 导入所有算法模块

# 通用问题算法
from .GA import GA
from .SA import SA

# 实数问题算法
from .DE import DE
from .PSO import PSO

# KP问题算法
from .DP_KP import DPKP
from .Greedy_KP import GreedyKP
from .NNDREAS import NNDREAS

# TSP问题算法
from .ACO import ACO
from .FI import FI
from .GFLS import GFLS
from .HGA_TSP import HGATSP



