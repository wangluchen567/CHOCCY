# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
枚举类型库
"""

import numpy as np
from enum import Enum, IntEnum
from typing import Dict, List


class VarType(IntEnum):
    """枚举问题变量的类型"""
    REAL = 1  # 实数
    INT = 2  # 整数
    BIN = 3  # 二进制
    PMU = 4  # 序列
    FIX = 5  # 固定标签

    @classmethod
    def validate(cls, value):
        """验证值是否为有效的变量类型"""
        return value in cls._value2member_map_

    @classmethod
    def _missing_(cls, value: int):
        """当找不到枚举值时自动调用"""
        if isinstance(value, (int, np.integer)):
            first = int(str(value)[0])
            if first in cls._value2member_map_:
                return cls(first)
        raise ValueError(f"{value} is not a valid VarType")

    @classmethod
    def convert(cls, values):
        """批量转换数组"""
        # 处理单个值
        if isinstance(values, (int, np.integer)):
            return cls(values)
        # 处理数组
        arr = np.asarray(values)
        # 向量化转换
        result = np.zeros_like(arr, dtype=int)
        for i, v in enumerate(arr.flat):
            result.flat[i] = cls(v).value
        return result

class VarTypeDict(dict):
    def __getitem__(self, key):
        if key in self:  # 精确匹配
            return super().__getitem__(key)
        return super().__getitem__(VarType(key))  # 回退到主类型


class StrEnum(Enum):
    """自定义字符串枚举，
    兼容 Python 3.11 以下版本"""

    @classmethod
    def _get_short_name_map(cls) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def parse(cls, value):
        """解析模式（仅支持字符串类型）"""
        if isinstance(value, cls):
            return value  # 若是本类型则直接返回
        if not isinstance(value, str):
            raise TypeError(f"Expected string, got {type(value).__name__}")
        # 统一小写处理
        value = value.lower().strip()
        # 先检查是否是有效的完整枚举值
        if value in cls._value2member_map_:
            return cls(value)
        # 检查是否是已知短名
        short_name_map = cls._get_short_name_map()
        if value in short_name_map:
            full_name = short_name_map[value]
            return cls(full_name)
        # 没有匹配项，抛出错误
        valid_options: List[str] = [member.value for member in cls]
        valid_options.sort()
        short_names: List[str] = list(short_name_map.keys())
        short_names.sort()
        raise ValueError(
            f"Unrecognized mode: '{value}'\n"
            f"Available full names: {', '.join(valid_options)}\n"
            f"Available short names: {', '.join(short_names)}"
        )

    @classmethod
    def validate(cls, value):
        """验证值是否为有效的变量类型"""
        try:
            cls.parse(value)
            return True
        except (ValueError, TypeError):
            return False


class VisualMode(StrEnum):
    """
    可视化模式枚举
    用于控制优化过程中的可视化输出方式。
    """
    NONE = 'none'  # 无可视化输出
    PROGRESS = 'progress'  # 进度条显示
    DECISIONS = 'decisions'  # 决策空间可视化
    OBJECTIVES = 'objectives'  # 目标空间可视化
    HYBRIDS_2D = 'hybrids_2d'  # 二维混合可视化
    HYBRIDS_3D = 'hybrids_3d'  # 三维混合可视化
    METRICS = 'metrics'  # 性能指标显示
    CUSTOM_PROBLEM = 'custom_problem'  # 问题特定可视化
    CUSTOM_ALGORITHM = 'custom_algorithm'  # 算法特定可视化
    LOG = 'log'  # 文本日志输出

    @classmethod
    def _get_short_name_map(cls) -> Dict[str, str]:
        """短名到完整枚举值的映射"""
        return {
            # 基础模式
            'off': 'none',
            # 进度
            'bar': 'progress',
            # 决策空间
            'dec': 'decisions',
            'decs': 'decisions',
            'x': 'decisions',
            'xs': 'decisions',
            # 目标空间
            'obj': 'objectives',
            'objs': 'objectives',
            'f': 'objectives',
            'fv': 'objectives',
            # 二维混合
            'h2d': 'hybrids_2d',
            'mix2d': 'hybrids_2d',
            # 三维混合
            'h3d': 'hybrids_3d',
            'mix3d': 'hybrids_3d',
            # 数据分析
            'metric': 'metrics',
            'score': 'metrics',
            'scores': 'metrics',
            # 自定义
            'cp': 'custom_problem',
            'pro': 'custom_problem',
            'prob': 'custom_problem',
            'problem': 'custom_problem',
            'ca': 'custom_algorithm',
            'alg': 'custom_algorithm',
            'algo': 'custom_algorithm',
            'algorithm': 'custom_algorithm',
        }


class MetricType(StrEnum):
    """
    监控指标类型枚举
    """
    HV = 'HV'  # 超体积指标
    GD = 'GD'  # 代际距离指标
    IGD = 'IGD'  # 逆代际距离指标
    GD_PLUS = 'GD+'  # 代际距离+指标
    IGD_PLUS = 'IGD+'  # 逆代际距离+指标
    PENALIZED_OBJ = 'Penalized Obj'  # 约束惩罚后的最优目标值

    @classmethod
    def _get_short_name_map(cls) -> Dict[str, str]:
        """短名到完整枚举值的映射"""
        return {
            'hv': 'HV',
            'gd': 'GD',
            'igd': 'IGD',
            'gd+': 'GD+',
            'igd+': 'IGD+',
            'pobj': 'Penalized Obj',
            'p_obj': 'Penalized Obj',
            'penalized obj': 'Penalized Obj',
        }


class AggregationMethod(StrEnum):
    """
    聚合方法类型
    """
    WSM = 'WeightedSum'  # 加权和法
    TCH = 'Tchebycheff'  # 切比雪夫法
    PBI = 'PBI'  # 基于惩罚的边界交叉法

    @classmethod
    def _get_short_name_map(cls) -> Dict[str, str]:
        """短名到完整枚举值的映射"""
        return {
            'wsm': 'WeightedSum',
            'sum': 'WeightedSum',
            'w_sum': 'WeightedSum',
            'weight_sum': 'WeightedSum',
            'weighted_sum': 'WeightedSum',
            'tchebycheff': 'Tchebycheff',
            'tch': 'Tchebycheff',
            'pbi': 'PBI',
        }
