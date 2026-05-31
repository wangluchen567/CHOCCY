# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
核心库
"""

import time
import warnings
from functools import wraps
from typing import Any, Callable

# ================ 数学常量 ================

INF = 1e10
NEG_INF = -1e10
EPS = 1e-12


# ================ 异常类 ================

class ChoccyError(Exception):
    """所有库异常的基类"""
    pass


class ProblemError(ChoccyError):
    """问题定义与配置相关错误"""
    pass


class SolutionError(ChoccyError):
    """解相关错误（单个解或解集合）"""
    pass


class AlgorithmError(ChoccyError):
    """算法执行与配置错误"""
    pass


class VisualizationError(ChoccyError):
    """可视化错误"""
    pass


# ============ 警告类 ============

class ChoccyWarning(Warning):
    """所有库警告的基类"""
    pass


class ProblemWarning(ChoccyWarning):
    """问题相关警告"""
    pass


class SolutionWarning(ChoccyWarning):
    """解相关警告"""
    pass


class AlgorithmWarning(ChoccyWarning):
    """算法执行相关警告"""
    pass


class PerformanceWarning(ChoccyWarning):
    """性能相关警告（使用慢速实现）"""
    pass


class VisualizationWarning(ChoccyWarning):
    """可视化警告"""
    pass


# ================ 工具函数 ================

def record_time(method: Callable) -> Callable:
    """统计实例方法的运行时间"""

    @wraps(method)  # 保留原始函数的元数据
    def timed(*args, **kwargs) -> Any:
        # 简单检查是否为实例方法
        if not args:
            raise TypeError("@record_time can only be used on instance methods")
        instance = args[0]
        # 检查是否有 run_times 属性
        if not hasattr(instance, 'run_time'):
            raise AttributeError(f"Class '{instance.__class__.__name__}' "
                                 f"must define 'run_time' attribute")
        # 记录时间
        start_time = time.perf_counter()
        result = method(*args, **kwargs)
        end_time = time.perf_counter()
        # 累加运行时间
        instance.run_time += (end_time - start_time)
        return result

    return timed


def warn_once(message, warning_class=Warning, stacklevel=2, key=None):
    """基于键值的单次警告"""
    if not hasattr(warn_once, '_warned_keys'):
        warn_once.warned_keys = set()

    # 使用key或message作为标识
    warning_key = key if key is not None else message

    if warning_key in warn_once.warned_keys:
        return False  # 已经警告过

    warnings.warn(message, warning_class, stacklevel=stacklevel)
    warn_once.warned_keys.add(warning_key)
    return True  # 首次警告


# 警告的显示级别
_WARNING_LEVEL = "default"


def set_warning_level(level="default"):
    """
    设置choccy警告的显示级别

    Parameters
    ----------
    level : str
        - "ignore": 忽略所有choccy警告
        - "default": 默认显示
        - "always": 总是显示
        - "error": 将警告转为异常
        - "verbose": 显示详细信息
    """
    global _WARNING_LEVEL
    _WARNING_LEVEL = level

    if level == "ignore":
        warnings.filterwarnings("ignore", category=ChoccyWarning)
    elif level == "always":
        warnings.filterwarnings("always", category=ChoccyWarning)
    elif level == "error":
        warnings.filterwarnings("error", category=ChoccyWarning)
    elif level == "verbose":
        warnings.filterwarnings("default", category=ChoccyWarning)
        warnings.simplefilter("once", ChoccyWarning)
    else:  # default
        warnings.filterwarnings("once", category=ChoccyWarning)


def get_warning_level():
    """获取当前警告级别"""
    return _WARNING_LEVEL


# 初始化
set_warning_level("default")
