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
class View:
    """可视化模式的静态参数"""
    NONE = -1  # 不显示
    BAR = 0  # 进度条
    OBJ = 1  # 目标空间
    DEC = 2  # 决策空间
    MIX2D = 3  # 二维混合空间
    MIX3D = 4  # 三维混合空间
    SCORE = 5  # 分数/指标
    PROB = 6  # 问题自定义绘图
    ALGO = 7  # 算法自定义绘图


# 导入所有父类
from .ALGORITHM import ALGORITHM
from .Comparator import Comparator
from .Evaluator import Evaluator
