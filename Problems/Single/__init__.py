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
# Demo问题
from .Ackley import Ackley
from .Sphere import Sphere

# SOP系列问题
from .SOP.SOP1 import SOP1
from .SOP.SOP2 import SOP2
from .SOP.SOP3 import SOP3
from .SOP.SOP4 import SOP4
from .SOP.SOP5 import SOP5
from .SOP.SOP6 import SOP6
from .SOP.SOP7 import SOP7
from .SOP.SOP8 import SOP8
from .SOP.SOP9 import SOP9
from .SOP.SOP10 import SOP10

# 实际问题
from .Practical.KP import KP  # 背包问题
from .Practical.TSP import TSP  # 旅行商问题
from .Practical.Regression import Regression  # 回归问题
from .Practical.Classification import Classification  # 分类问题
from .Practical.FixLabelCluster import FixLabelCluster  # 固定数量聚类问题
from .Practical.MixFixLabelCluster import MixFixLabelCluster  # 混合变量聚类问题
