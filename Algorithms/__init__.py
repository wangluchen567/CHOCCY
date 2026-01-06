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
    NONE = 'none'  # 不显示
    BAR = 'bar'  # 进度条
    OBJ = 'obj'  # 目标空间
    DEC = 'dec'  # 决策空间
    MIX2D = 'mix2d'  # 二维混合空间
    MIX3D = 'mix3d'  # 三维混合空间
    SCORE = 'score'  # 分数/指标
    PROB = 'problem'  # 问题自定义绘图
    ALGO = 'algorithm'  # 算法自定义绘图
    LOG = 'log'  # 输出日志


# 导入所有父类
from .ALGORITHM import ALGORITHM
from .Comparator import Comparator
from .Evaluator import Evaluator
