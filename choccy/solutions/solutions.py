# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
解集父类
"""

import warnings
import numpy as np
from typing import Optional, Union, Dict, Callable
from ..core import SolutionError, SolutionWarning
from ..utilities.commons.sorting import fast_nd_sort
from ..utilities.handler import save_to_file, load_from_file


class Solutions(object):

    def __init__(self,
                 decs: Union[np.ndarray, list],
                 objs: Optional[Union[np.ndarray, list]] = None,
                 cons: Optional[Union[np.ndarray, list]] = None,
                 fits: Optional[Union[np.ndarray, list]] = None,
                 metrics: Optional[Dict[str, float]] = None):
        """
        解集合类

        :param decs: 决策变量矩阵，形状为 (n_sols, n_vars)
        :param objs: 目标值矩阵，第一维形状需与 decs 匹配，形状为 (n_sols, n_objs)
        :param cons: 约束值矩阵，第一维形状需与 decs 匹配，形状为 (n_sols, n_cons)
        :param fits: 适应度值向量，第一维形状需与 decs 匹配，形状为 (n_sols,)
        :param metrics: 指标字典（key: 字符串类型, value: 浮点类型）
        """
        # 使用私有属性存储数据
        self._decs = None
        self._objs = None
        self._cons = None
        self._fits = None
        # 初始化指标字典
        self._metrics: Dict[str, float] = {}
        # 调整函数
        self._decs_func: Optional[Callable] = None
        # 评估函数
        self._objs_func: Optional[Callable] = None
        self._cons_func: Optional[Callable] = None
        self._fits_func: Optional[Callable] = None
        self._metric_funcs: Dict[str, Callable] = {}
        # 初始化数据（通过setter确保正确性）
        self.decs = decs
        if objs is not None:
            self.objs = objs
        if cons is not None:
            self.cons = cons
        if fits is not None:
            self.fits = fits
        if metrics is not None:
            self.metrics = metrics
        # 验证形状一致性
        self._validate_shapes()

    @property
    def decs(self) -> np.ndarray:
        """获取决策变量矩阵"""
        return self._decs

    @decs.setter
    def decs(self, value: Union[np.ndarray, list]):
        """设置决策变量矩阵"""
        # 转换为正确格式
        value_arr = self.to_row(np.asarray(value, dtype=float))
        # 如果之前有数据，验证维度一致性
        if self._decs is not None and value_arr.shape[1] != self.n_vars:
            raise SolutionError(
                f"Number of decision variables mismatch: "
                f"new has {value_arr.shape[1]}, current has {self.n_vars}"
            )
        # 设置新值
        self._decs = value_arr
        # 处理objs
        if self._objs is not None:
            self._objs = self._create_empty_like(self._objs, value_arr.shape[0])
        # 处理cons
        if self._cons is not None:
            self._cons = self._create_empty_like(self._cons, value_arr.shape[0])
        # 处理fits
        if self._fits is not None:
            self._fits = self._create_empty_like(self._fits, value_arr.shape[0])

    @property
    def objs(self) -> Optional[np.ndarray]:
        """获取目标值矩阵"""
        return self._objs

    @objs.setter
    def objs(self, value: Union[np.ndarray, list, None]):
        """设置目标值矩阵"""
        if value is None:
            self._objs = None
            return
        # 转换为二维数组
        value_arr = self.to_row(np.asarray(value, dtype=float))
        # 验证形状
        if value_arr.shape[0] != self.n_sols:
            raise SolutionError(
                f"objs shape {value_arr.shape} does not match "
                f"number of solutions {self.n_sols}"
            )
        # 赋值数据
        self._objs = value_arr

    @property
    def cons(self) -> Optional[np.ndarray]:
        """获取约束值矩阵"""
        return self._cons

    @cons.setter
    def cons(self, value: Union[np.ndarray, list, None]):
        """设置约束值矩阵"""
        if value is None:
            self._cons = None
            return
        # 转换为二维数组
        value_arr = self.to_row(np.asarray(value, dtype=float))
        # 验证形状
        if value_arr.shape[0] != self.n_sols:
            raise SolutionError(
                f"cons shape {value_arr.shape} does not match "
                f"number of solutions {self.n_sols}"
            )
        # 赋值数据
        self._cons = value_arr

    @property
    def fits(self) -> Optional[np.ndarray]:
        """获取适应度值向量"""
        return self._fits

    @fits.setter
    def fits(self, value: Union[np.ndarray, list, None]):
        """设置适应度值向量"""
        if value is None:
            self._fits = None
            return
        # 转换为一维数组
        value_arr = np.asarray(value, dtype=float).flatten()
        # 验证形状
        if value_arr.shape[0] != self.n_sols:
            raise SolutionError(
                f"fits shape {value_arr.shape} does not match "
                f"number of solutions {self.n_sols}"
            )
        # 赋值数据
        self._fits = value_arr

    @property
    def metrics(self) -> Dict[str, float]:
        """获取指标字典"""
        return self._metrics

    @metrics.setter
    def metrics(self, value: Optional[Dict[str, float]]) -> None:
        """设置指标字典"""
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"metrics must be dict, got {type(value)}")
        self._metrics = value if value is not None else {}

    def _validate_shapes(self):
        """验证 decs, objs, cons 和 fits 的数量一致性"""
        if self._objs is not None and self._objs.shape[0] != self.n_sols:
            raise SolutionError(
                f"objs shape {self._objs.shape} does not match "
                f"number of solutions {self.n_sols}"
            )
        if self._cons is not None and self._cons.shape[0] != self.n_sols:
            raise SolutionError(
                f"cons shape {self._cons.shape} does not match "
                f"number of solutions {self.n_sols}"
            )
        if self._fits is not None and self._fits.shape[0] != self.n_sols:
            raise SolutionError(
                f"fits shape {self._fits.shape} does not match "
                f"number of solutions {self.n_sols}"
            )

    @property
    def xs(self):
        """decs 的简短别名"""
        return self.decs

    @xs.setter
    def xs(self, value):
        """通过xs设置decs"""
        self.decs = value

    @property
    def fv(self):
        """objs 的简短别名"""
        return self.objs

    @fv.setter
    def fv(self, value):
        """通过fv设置objs"""
        self.objs = value

    @property
    def cv(self):
        """cons 的简短别名"""
        return self.cons

    @cv.setter
    def cv(self, value):
        """通过cv设置cons"""
        self.cons = value

    @property
    def ft(self):
        """fits 的简短别名"""
        return self.fits

    @ft.setter
    def ft(self, value):
        """通过ft设置fits"""
        self.fits = value

    @property
    def x(self):
        """获取简化 decs"""
        return self._simplify(self.decs)

    @property
    def f(self):
        """获取简化 objs"""
        return self._simplify(self.objs)

    @property
    def c(self):
        """获取简化 cons"""
        return self._simplify(self.cons)

    @property
    def t(self):
        """获取简化 fits"""
        return self._simplify(self.fits)

    @property
    def n_sols(self) -> int:
        """解的数量"""
        return self.decs.shape[0]

    @property
    def n_vars(self) -> int:
        """决策变量个数"""
        return self.decs.shape[1]

    @property
    def n_decs(self) -> int:
        """决策变量个数 (别称)"""
        return self.n_vars

    @property
    def n_objs(self) -> Optional[int]:
        """目标个数"""
        return self.objs.shape[1] if self.objs is not None else None

    @property
    def n_cons(self) -> Optional[int]:
        """约束个数"""
        return self.cons.shape[1] if self.cons is not None else None

    @property
    def is_single(self) -> bool:
        """是否只包含单个解"""
        return self.n_sols == 1

    @property
    def hv(self) -> Optional[float]:
        """获取超体积指标（如果存在）"""
        return self.get_metric('HV')

    @property
    def gd(self) -> Optional[float]:
        """获取gd指标（如果存在）"""
        return self.get_metric('GD')

    @property
    def gd_plus(self) -> Optional[float]:
        """获取gd指标（如果存在）"""
        return self.get_metric('GD+')

    @property
    def igd(self) -> Optional[float]:
        """获取igd指标（如果存在）"""
        return self.get_metric('IGD')

    @property
    def igd_plus(self) -> Optional[float]:
        """获取igd+指标（如果存在）"""
        return self.get_metric('IGD+')

    def __len__(self) -> int:
        return self.n_sols

    def __getitem__(self, key) -> 'Solutions':
        """索引/切片接口"""
        if isinstance(key, np.ndarray) and key.dtype == bool:
            if key.shape[0] != self.n_sols:
                raise IndexError(
                    f"Boolean index length {key.shape[0]} "
                    f"does not match Solutions length {self.n_sols}"
                )
            if not key.any():  # 全False，返回空解集
                return Solutions(decs=np.empty((0, self.n_vars)))
            # 转换为整数索引，与__setitem__保持一致
            indices = np.where(key)[0]
        elif isinstance(key, (list, np.ndarray)) and len(key) == 0:
            # 空列表/空数组，返回空解集
            return Solutions(decs=np.empty((0, self.n_vars)))
        else:
            # 其他索引类型（int, slice等）
            indices = key
        # 切片数据
        decs_slice = self.decs[indices]
        objs_slice = None if self.objs is None else self.objs[indices]
        cons_slice = None if self.cons is None else self.cons[indices]
        fits_slice = None if self.fits is None else self.fits[indices]
        # 确保至少2维（除了fits）
        if decs_slice.ndim == 1:
            decs_slice = self.to_row(decs_slice)
        if objs_slice is not None and objs_slice.ndim == 1:
            objs_slice = self.to_row(objs_slice)
        if cons_slice is not None and cons_slice.ndim == 1:
            cons_slice = self.to_row(cons_slice)
        # fits保持一维
        if fits_slice is not None and fits_slice.ndim == 0:
            fits_slice = np.array([fits_slice])
        # 创建新对象
        solutions = Solutions(
            decs=decs_slice,
            objs=objs_slice,
            cons=cons_slice,
            fits=fits_slice,
            # 设置指标为nan
            metrics=dict.fromkeys(self.metrics, float('nan'))
        )
        # 复制调整与评估函数
        solutions._decs_func = self._decs_func
        solutions._objs_func = self._objs_func
        solutions._cons_func = self._cons_func
        solutions._fits_func = self._fits_func
        solutions._metric_funcs = self._metric_funcs.copy()
        # 返回新对象
        return solutions

    def __setitem__(self, key, value: 'Solutions'):
        """支持索引/切片赋值"""
        if not isinstance(value, Solutions):
            raise TypeError("Can only assign Solutions objects")
        # 处理不同类型的key
        if isinstance(key, np.ndarray) and key.dtype == bool:
            # 全False布尔数组直接返回
            if not key.any():
                return
            # 布尔掩码
            if key.shape[0] != self.n_sols:
                raise ValueError(
                    f"Boolean mask length mismatch: {key.shape[0]} != {self.n_sols}"
                )
            n_selected = np.sum(key)
            if n_selected != value.n_sols:
                raise SolutionError(
                    f"Mask selects {n_selected} positions, "
                    f"but {value.n_sols} solutions provided"
                )
            # 转换为整数索引进行统一处理
            indices = np.where(key)[0]
            value_indices = slice(None)
        elif isinstance(key, int):
            # 单个索引赋值：sols[0] = other_sol
            indices = [key]
            value_indices = [0] if value.n_sols == 1 else slice(None)
        elif isinstance(key, slice):
            # 切片赋值：sols[1:3] = other_slice
            indices = range(*key.indices(self.n_sols))
            value_indices = slice(None)
        elif isinstance(key, (list, np.ndarray)):
            # 空列表/空数组则直接返回
            if len(key) == 0:
                return
            # 列表/数组索引赋值(非布尔数组)
            indices = key
            value_indices = slice(None)
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")
        # 若无修改索引则直接返回
        if len(indices) == 0:
            return
        # 验证数量和维度匹配
        if len(indices) != value.n_sols:
            # 广播情况：右侧只有1个解，左侧有多个位置
            if value.n_sols == 1 and len(indices) > 1:
                # 将value_indices固定为0，后续所有位置都赋这个解
                value_indices = 0
            else:
                raise ValueError(
                    f"Number of indices ({len(indices)}) does not match "
                    f"number of solutions to assign ({value.n_sols})"
                )
        if value.n_vars != self.n_vars:
            raise ValueError(
                f"Number of decision variables mismatch: {value.n_vars} != {self.n_vars}"
            )
        # 赋值决策变量矩阵
        self.decs[indices] = value.decs[value_indices]
        # 赋值目标值矩阵
        if self._objs is not None and value.objs is not None:
            self._objs[indices] = value.objs[value_indices]
        elif self._objs is not None and value.objs is None:
            # 如果被赋值的目标值为None，清空对应位置
            self._objs[indices] = np.nan
        # 赋值约束值矩阵
        if self._cons is not None and value.cons is not None:
            self._cons[indices] = value.cons[value_indices]
        elif self._cons is not None and value.cons is None:
            # 如果被赋值的目标值为None，清空对应位置
            self._cons[indices] = np.nan
        # 赋值适应度值
        if self._fits is not None and value.fits is not None:
            self._fits[indices] = value.fits[value_indices]
        elif self._fits is not None and value.fits is None:
            # 如果被赋值的适应度为None，清空对应位置
            self._fits[indices] = np.nan
        # 处理metrics（标记为无效指标 NaN）
        self.metrics = dict.fromkeys(self.metrics, float('nan'))

    def get(self, idx: Union[np.ndarray, list, int, slice]) -> 'Solutions':
        """获取单个解或切片"""
        return self[idx]

    def dec(self, idx: int, copy: bool = True) -> np.ndarray:
        """获取第idx个决策向量"""
        if idx >= self.n_sols:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sols})")
        if copy:  # 返回副本（安全，外部不可修改）
            return self.decs[idx].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.decs[idx]

    def decs_mat(self, indices: Union[int, slice, list, np.ndarray], copy: bool = True) -> np.ndarray:
        """获取决策矩阵"""
        if isinstance(indices, int):
            indices = [indices]
        if copy:  # 返回副本（安全，外部不可修改）
            return self.decs[indices].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.decs[indices]

    def obj(self, idx: int, copy: bool = True) -> Optional[np.ndarray]:
        """获取第idx个目标向量"""
        if self.objs is None:
            return None
        if idx >= self.n_sols:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sols})")
        if copy:  # 返回副本（安全，外部不可修改）
            return self.objs[idx].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.objs[idx]

    def objs_mat(self, indices: Union[int, slice, list, np.ndarray], copy: bool = True) -> Optional[np.ndarray]:
        """获取目标矩阵"""
        if self.objs is None:
            return None
        if isinstance(indices, int):
            indices = [indices]
        if copy:  # 返回副本（安全，外部不可修改）
            return self.objs[indices].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.objs[indices]

    def con(self, idx: int, copy: bool = True) -> Optional[np.ndarray]:
        """获取第idx个约束向量"""
        if self.cons is None:
            return None
        if idx >= self.n_sols:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sols})")
        if copy:  # 返回副本（安全，外部不可修改）
            return self.cons[idx].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.cons[idx]

    def cons_mat(self, indices: Union[int, slice, list, np.ndarray], copy: bool = True) -> Optional[np.ndarray]:
        """获取约束矩阵"""
        if self.cons is None:
            return None
        if isinstance(indices, int):
            indices = [indices]
        if copy:  # 返回副本（安全，外部不可修改）
            return self.cons[indices].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.cons[indices]

    def fit(self, idx: int, copy: bool = True) -> Optional[float]:
        """获取第idx个适应度值"""
        if self.fits is None:
            return None
        if idx >= self.n_sols:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sols})")
        if copy:  # 返回副本（安全，外部不可修改）
            return float(self.fits[idx].copy())
        else:  # 返回源数据（不安全，可被外部修改）
            return float(self.fits[idx])

    def fits_vec(self, indices: Union[int, slice, list, np.ndarray], copy: bool = True) -> Optional[np.ndarray]:
        """获取适应度向量"""
        if self.fits is None:
            return None
        if isinstance(indices, int):
            indices = [indices]
        if copy:  # 返回副本（安全，外部不可修改）
            return self.fits[indices].copy()
        else:  # 返回源数据（不安全，可被外部修改）
            return self.fits[indices]

    def set_metric(self, name: str, value: Union[float, int]) -> 'Solutions':
        """设置/更新指标值"""
        self._metrics[name] = float(value)
        return self

    def add_metric(self, name: str, value: Union[float, int]) -> 'Solutions':
        """添加指标值，如果已存在则覆盖"""
        if name in self._metrics:
            warnings.warn(
                f"Metric '{name}' already exists, will be overwritten",
                SolutionWarning,
                stacklevel=2
            )
        return self.set_metric(name, value)

    def remove_metric(self, name: str) -> 'Solutions':
        """移除指标"""
        if name in self._metrics:
            del self._metrics[name]
        return self

    def get_metric(self, name: str, default: float = float('nan')) -> float:
        """获取指标值"""
        return self._metrics.get(name, default)

    def has_metric(self, name: str) -> bool:
        """检查是否存在指定指标"""
        return name in self._metrics

    def set_decs_func(self, func: Callable) -> 'Solutions':
        """设置决策变量调整函数"""
        self._decs_func = func
        return self

    def set_objs_func(self, func: Callable) -> 'Solutions':
        """设置目标计算函数"""
        self._objs_func = func
        return self

    def set_cons_func(self, func: Callable) -> 'Solutions':
        """设置约束计算函数"""
        self._cons_func = func
        return self

    def set_fits_func(self, func: Callable) -> 'Solutions':
        """设置适应度计算函数"""
        self._fits_func = func
        return self

    def set_metric_func(self, name: str, func: Callable) -> 'Solutions':
        """设置指标计算函数"""
        self._metric_funcs[name] = func
        return self

    def set_eval_funcs(self,
                       decs_func: Optional[Callable] = None,
                       objs_func: Optional[Callable] = None,
                       cons_func: Optional[Callable] = None,
                       fits_func: Optional[Callable] = None,
                       metric_funcs: Optional[Dict[str, Callable]] = None) -> 'Solutions':
        """批量设置评估函数"""
        if decs_func is not None:
            self.set_decs_func(decs_func)
        if objs_func is not None:
            self.set_objs_func(objs_func)
        if cons_func is not None:
            self.set_cons_func(cons_func)
        if fits_func is not None:
            self.set_fits_func(fits_func)
        if metric_funcs is not None:
            for name, func in metric_funcs.items():
                self.set_metric_func(name, func)
        # 返回自身，方便链式调用
        return self

    def adjust_decs(self) -> 'Solutions':
        """调整决策变量"""
        if self._decs_func is None:
            return self  # 静默返回，不做任何调整
        try:
            # 传入self，允许调整函数访问所有属性
            result = self._decs_func(self)
            self.decs = self.to_row(result)
        except Exception as e:
            warnings.warn(f"decs function failed: {e}", SolutionWarning, stacklevel=2)
        # 返回自身，方便链式调用
        return self


    def eval_objs(self) -> 'Solutions':
        """评估目标值"""
        if self._objs_func is None:
            warnings.warn("Objective function not set", SolutionWarning, stacklevel=2)
            return self
        try:
            # 传入self，允许评估函数访问所有属性
            result = self._objs_func(self)
            self.objs = self.to_row(result)
        except Exception as e:
            warnings.warn(f"Objective function evaluation failed: {e}", SolutionWarning, stacklevel=2)
        # 返回自身，方便链式调用
        return self

    def eval_cons(self) -> 'Solutions':
        """评估约束值"""
        if self._cons_func is None:
            warnings.warn("Constraint function not set", SolutionWarning, stacklevel=2)
            return self
        try:
            # 传入self，允许约束函数访问所有属性
            result = self._cons_func(self)
            self.cons = self.to_row(result)
        except Exception as e:
            warnings.warn(f"Constraint function evaluation failed: {e}", SolutionWarning, stacklevel=2)
        # 返回自身，方便链式调用
        return self

    def eval_fits(self) -> 'Solutions':
        """评估适应度值"""
        if self._fits_func is None:
            warnings.warn("fits function not set", SolutionWarning, stacklevel=2)
            return self
        try:
            # 传入self，允许适应度函数访问所有属性
            result = self._fits_func(self)
            # 确保结果为一维数组
            result = np.asarray(result, dtype=float).flatten()
            self.fits = result
        except Exception as e:
            warnings.warn(f"fits function evaluation failed: {e}", SolutionWarning, stacklevel=2)
        # 返回自身，方便链式调用
        return self

    def eval_metrics(self) -> 'Solutions':
        """仅评估指标（假设基础数据已计算）"""
        for name, func in self._metric_funcs.items():
            try:
                metric_value = func(self)
                # 确保指标为浮点数
                if isinstance(metric_value, (np.ndarray, list)):
                    metric_value = float(np.asarray(metric_value).item())
                self.set_metric(name, metric_value)
            except Exception as e:
                warnings.warn(f"Metric '{name}' calculation failed: {e}", SolutionWarning, stacklevel=2)
        # 返回自身，方便链式调用
        return self

    def evaluate(self) -> 'Solutions':
        """执行评估(评估基础信息，不包含指标)"""
        # 评估目标值
        self.eval_objs()
        # 评估约束值
        self.eval_cons()
        # 评估适应度值
        self.eval_fits()
        # 返回自身，方便链式调用
        return self

    def evaluate_full(self) -> 'Solutions':
        """执行完整评估(包含指标)"""
        # 评估基础信息
        self.evaluate()
        # 评估指标值
        self.eval_metrics()
        # 返回自身，方便链式调用
        return self

    def concat(self, other: 'Solutions', ignore_warn: bool = True) -> 'Solutions':
        """合并另一个Solutions对象"""
        if other.n_vars != self.n_vars:
            raise ValueError(
                f"Number of decision variables mismatch: {self.n_vars} != {other.n_vars}"
            )
        # 合并决策变量矩阵
        new_decs = np.vstack([self.decs, other.decs])
        # 合并目标值矩阵
        new_objs = None
        if self.objs is not None and other.objs is not None:
            new_objs = np.vstack([self.objs, other.objs])
        elif self.objs is not None or other.objs is not None:
            if not ignore_warn:
                warnings.warn(
                    "objs status inconsistent between two Solutions objects",
                    SolutionWarning,
                    stacklevel=2
                )
        # 合并约束值矩阵
        new_cons = None
        if self.cons is not None and other.cons is not None:
            new_cons = np.vstack([self.cons, other.cons])
        elif self.cons is not None or other.cons is not None:
            if not ignore_warn:
                warnings.warn(
                    "cons status inconsistent between two Solutions objects",
                    SolutionWarning,
                    stacklevel=2
                )
        # 合并适应度值
        new_fits = None
        if self.fits is not None and other.fits is not None:
            new_fits = np.concatenate([self.fits, other.fits])
        elif self.fits is not None or other.fits is not None:
            if not ignore_warn:
                warnings.warn(
                    "fits status inconsistent between two Solutions objects",
                    SolutionWarning,
                    stacklevel=2
                )
        # 合并指标 - 对于合并操作，指标可能需要重新计算
        # 只保留在两者中都存在的指标
        common_metrics = [key for key in self.metrics.keys() if key in other.metrics]
        if not ignore_warn:
            for name in common_metrics:
                warnings.warn(
                    f"Metric '{name}' exists in both objects, "
                    f"you may need to recalculate it after merging",
                    SolutionWarning,
                    stacklevel=2
                )
        # 创建新对象
        solutions = Solutions(
            decs=new_decs,
            objs=new_objs,
            cons=new_cons,
            fits=new_fits,
            # 所有指标均置为 nan，因为需要重新计算
            metrics=dict.fromkeys(common_metrics, float('nan'))
        )
        # 合并调整与评估函数（以第一个为主）
        solutions._decs_func = self._decs_func
        solutions._objs_func = self._objs_func
        solutions._cons_func = self._cons_func
        solutions._fits_func = self._fits_func
        solutions._metric_funcs = {**self._metric_funcs, **other._metric_funcs}
        # 返回新对象
        return solutions

    def copy(self) -> 'Solutions':
        """创建深拷贝"""
        solutions = Solutions(
            decs=self.decs.copy(),
            objs=self.objs.copy() if self.objs is not None else None,
            cons=self.cons.copy() if self.cons is not None else None,
            fits=self.fits.copy() if self.fits is not None else None,
            metrics=self.metrics.copy() if self.metrics is not None else None
        )
        # 复制调整与评估函数
        solutions._decs_func = self._decs_func
        solutions._objs_func = self._objs_func
        solutions._cons_func = self._cons_func
        solutions._fits_func = self._fits_func
        solutions._metric_funcs = self._metric_funcs.copy()
        # 返回复制对象
        return solutions

    def shuffle(self, inplace: bool = True, seed: Optional[int] = None) -> Optional['Solutions']:
        """
        随机打乱所有解（保持决策变量、目标值、约束值、适应度、指标的对应关系）
        :param inplace: 是否原地打乱
        :param seed: 随机种子
        :return: 如果inplace=False，返回新的Solutions对象；否则返回None
        """
        # 只有一个解或没有解，不需要打乱
        if self.n_sols <= 1:
            if not inplace:
                return self.copy()
            return None
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random
        # 生成随机排列索引
        shuffle_indices = rng.permutation(self.n_sols)
        if inplace:
            # 原地打乱
            self._shuffle_inplace(shuffle_indices)
            return None
        else:
            # 返回新的Solutions对象
            return self._shuffle_copy(shuffle_indices)

    def _shuffle_inplace(self, indices: np.ndarray):
        """原地打乱（修改当前对象）"""
        # 打乱决策变量矩阵
        self._decs = self._decs[indices]
        # 打乱目标值矩阵（如果存在）
        if self._objs is not None:
            self._objs = self._objs[indices]

        # 打乱约束值矩阵（如果存在）
        if self._cons is not None:
            self._cons = self._cons[indices]

        # 打乱适应度值（如果存在）
        if self._fits is not None:
            self._fits = self._fits[indices]

    def _shuffle_copy(self, indices: np.ndarray) -> 'Solutions':
        """返回打乱后的新对象（不修改原对象）"""
        # 创建打乱后的副本
        shuffled_decs = self.decs[indices]
        # 创建打乱后的副本
        shuffled_objs = None
        if self.objs is not None:
            shuffled_objs = self.objs[indices]
        # 创建打乱后的副本
        shuffled_cons = None
        if self.cons is not None:
            shuffled_cons = self.cons[indices]
        # 创建打乱后的副本
        shuffled_fits = None
        if self.fits is not None:
            shuffled_fits = self.fits[indices]
        # 创建新的Solutions对象
        solutions = Solutions(
            decs=shuffled_decs,
            objs=shuffled_objs,
            cons=shuffled_cons,
            fits=shuffled_fits,
            metrics=self.metrics.copy()
        )
        # 复制调整与评估函数
        solutions._decs_func = self._decs_func
        solutions._objs_func = self._objs_func
        solutions._cons_func = self._cons_func
        solutions._fits_func = self._fits_func
        solutions._metric_funcs = self._metric_funcs.copy()
        # 返回打乱后的对象
        return solutions

    def get_best(self) -> 'Solutions':
        """
        获取最优解集合（单目标返回一个解，多目标返回Pareto前沿）

        选择策略：
        1. 优先选择满足约束的解（cons <= 0）
        2. 如果没有可行解，选择约束违反最小的解
        3. 单目标：选择目标值最小的解
        4. 多目标：返回整个Pareto前沿
        :return: 返回一个新的Solutions对象
        """
        if self.objs is None:
            raise SolutionError("Cannot get best solutions: objectives not evaluated")

        # 检查是否存在约束
        has_cons = (self.cons is not None and self.n_cons > 0)

        # 判断是否满足约束
        if has_cons:
            # cons可能是多列的，需要检查所有列
            feas = np.all(self.cons <= 0, axis=1)
            decs_feas = self.decs[feas]
            objs_feas = self.objs[feas]
            cons_feas = self.cons[feas] if self.cons is not None else None
            fits_feas = self.fits[feas] if self.fits is not None else None
        else:
            # 没有约束，所有解都是可行的
            decs_feas = self.decs
            objs_feas = self.objs
            cons_feas = None
            fits_feas = self.fits

        # 如果没有可行解（且存在约束），选择约束违反最小的解
        if has_cons and len(decs_feas) == 0:
            # 计算约束违反程度
            violation = np.sum(np.maximum(self.cons, 0), axis=1)
            best_indices = [np.argmin(violation)]
            best_decs = self.decs[best_indices]
            best_objs = self.objs[best_indices] if self.objs is not None else None
            best_cons = self.cons[best_indices] if self.cons is not None else None
            best_fits = self.fits[best_indices] if self.fits is not None else None
        else:
            # 有可行解的情况
            if self.n_objs == 1:
                # 单目标：选择目标值最小的解
                min_index = np.argmin(objs_feas)
                best_indices = [min_index]
            elif self.n_objs > 1:
                # 多目标：使用快速非支配排序找到Pareto前沿
                # 对可行解进行非支配排序
                fronts, _ = fast_nd_sort(objs_feas)
                # 第一前沿就是Pareto前沿
                best_indices = fronts[0]
            else:
                raise SolutionError(
                    f"Invalid number of objectives: {self.n_objs}. "
                    f"Objectives must be >= 1. Please check your problem definition."
                )
            # 创建最优解的Solutions对象
            best_decs = decs_feas[best_indices]
            best_objs = objs_feas[best_indices] if objs_feas is not None else None
            best_cons = cons_feas[best_indices] if cons_feas is not None else None
            best_fits = fits_feas[best_indices] if fits_feas is not None else None

        # 创建新的Solutions对象
        best_solutions = Solutions(
            decs=best_decs,
            objs=best_objs,
            cons=best_cons,
            fits=best_fits,
            metrics=self.metrics.copy()
        )
        # 复制调整与评估函数
        best_solutions._decs_func = self._decs_func
        best_solutions._objs_func = self._objs_func
        best_solutions._cons_func = self._cons_func
        best_solutions._fits_func = self._fits_func
        best_solutions._metric_funcs = self._metric_funcs.copy()
        # 返回最优解(集合)
        return best_solutions

    def create_empty(self) -> 'Solutions':
        """创建与当前实例形状相同的未初始化副本"""
        # 创建未初始化的decs
        empty_decs = np.empty_like(self.decs)
        empty_decs[:] = np.nan  # 赋值为nan
        # 创建objs（如果当前有）
        empty_objs = None
        if self.objs is not None:
            empty_objs = self._create_empty_like(self.objs)
        # 创建cons（如果当前有）
        empty_cons = None
        if self.cons is not None:
            empty_cons = self._create_empty_like(self.cons)
        # 创建fits（如果当前有）
        empty_fits = None
        if self.fits is not None:
            empty_fits = self._create_empty_like(self.fits)
        # 创建新实例
        new_solutions = Solutions(
            decs=empty_decs,
            objs=empty_objs,
            cons=empty_cons,
            fits=empty_fits,
            # 所有指标均置为 nan
            metrics=dict.fromkeys(self.metrics, float('nan'))
        )
        # 复制调整与评估函数
        new_solutions._decs_func = self._decs_func
        new_solutions._objs_func = self._objs_func
        new_solutions._cons_func = self._cons_func
        new_solutions._fits_func = self._fits_func
        new_solutions._metric_funcs = self._metric_funcs.copy()
        # 返回新的空解
        return new_solutions

    def create_new_with(self,
                        decs: Optional[np.ndarray] = None,
                        objs: Optional[np.ndarray] = None,
                        cons: Optional[np.ndarray] = None,
                        fits: Optional[np.ndarray] = None) -> 'Solutions':
        """
        基于当前实例创建新实例，可指定要替换的部分
        :param decs: 新决策变量矩阵（可选）
        :param objs: 新目标值矩阵（可选）
        :param cons: 新约束值矩阵（可选）
        :param fits: 新适应度值（可选）
        :return: 新实例解
        """
        # 确定解数量
        if decs is not None:
            new_decs = self.to_row(np.asarray(decs, dtype=float))
            n_sols = new_decs.shape[0]
        else:
            n_sols = self.n_sols
            # 未提供decs，创建全NaN
            new_decs = self._create_empty_like(self.decs)
        # 处理目标值矩阵
        if objs is not None:
            new_objs = self.to_row(np.asarray(objs, dtype=float))
            if new_objs.shape[0] != n_sols:
                raise ValueError(
                    f"objs has {new_objs.shape[0]} solutions, "
                    f"but decs has {n_sols} solutions"
                )
        elif self.objs is not None:
            # 未提供objs，创建全NaN（保持相同目标数）
            new_objs = self._create_empty_like(self.objs)
        else:
            new_objs = None
        # 处理约束值矩阵
        if cons is not None:
            new_cons = self.to_row(np.asarray(cons, dtype=float))
            if new_cons.shape[0] != n_sols:
                raise ValueError(
                    f"cons has {new_cons.shape[0]} solutions, "
                    f"but decs has {n_sols} solutions"
                )
        elif self.cons is not None:
            # 未提供cons，创建全NaN（保持相同约束数）
            new_cons = self._create_empty_like(self.cons)
        else:
            new_cons = None
        # 处理适应度值
        if fits is not None:
            new_fits = np.asarray(fits, dtype=float).flatten()
            if new_fits.shape[0] != n_sols:
                raise ValueError(
                    f"fits has {new_fits.shape[0]} solutions, "
                    f"but decs has {n_sols} solutions"
                )
        elif self.fits is not None:
            # 未提供fits，创建全NaN（保持相同形状）
            new_fits = self._create_empty_like(self.fits)
        else:
            new_fits = None
        # 创建新实例
        new_solutions = Solutions(
            decs=new_decs,
            objs=new_objs,
            cons=new_cons,
            fits=new_fits,
            # 所有指标都置为 nan
            metrics=dict.fromkeys(self.metrics, float('nan'))
        )
        # 复制调整与评估函数
        new_solutions._decs_func = self._decs_func
        new_solutions._objs_func = self._objs_func
        new_solutions._cons_func = self._cons_func
        new_solutions._fits_func = self._fits_func
        new_solutions._metric_funcs = self._metric_funcs.copy()
        # 返回新解
        return new_solutions

    def save(self, file_path: str, file_format: Optional[str] = None, as_object: bool = False) -> None:
        """
        保存Solutions到文件
        :param file_path: 文件路径
        :param file_format: 格式 'csv', 'json', 'pkl', 'npz'
        :param as_object: 仅对pkl有效，True保存完整对象，False保存dict
        """
        if file_format == 'pkl' and as_object:
            save_to_file(self, file_path, file_format)
        else:
            data = {
                'decs': self._decs.tolist() if self._decs is not None else None,
                'objs': self._objs.tolist() if self._objs is not None else None,
                'cons': self._cons.tolist() if self._cons is not None else None,
                'fits': self._fits.tolist() if self._fits is not None else None,
                'metrics': self.metrics
            }
            save_to_file(data, file_path, file_format)

    @classmethod
    def load(cls, file_path: str, file_format: Optional[str] = None) -> 'Solutions':
        """
        从文件加载Solutions
        :param file_path: 文件路径
        :param file_format: 格式 'csv', 'json', 'pkl', 'npz', 可为空以自动识别
        :return: Solutions对象
        """
        data = load_from_file(file_path, file_format)

        if isinstance(data, Solutions):
            return data

        return cls(
            decs=data.get('decs'),
            objs=data.get('objs'),
            cons=data.get('cons'),
            fits=data.get('fits'),
            metrics=data.get('metrics', {})
        )

    @staticmethod
    def _simplify(arr: Optional[np.ndarray]) -> Optional[Union[float, np.ndarray]]:
        """将数组转换为更简洁的形式，去除不必要的维度"""
        if arr is None or arr.size == 0:
            return None  # 空数组
        if arr.size == 1:  # 单元素
            return float(arr.flat[0])
        if arr.ndim == 2 and arr.shape[0] == 1:  # 单行
            return arr.flatten()
        return arr.copy()

    @staticmethod
    def to_row(x: np.ndarray) -> np.ndarray:
        """将输入转换为行向量（形状为 (1, n)）"""
        return x.reshape(1, -1) if x.ndim <= 1 else x

    @staticmethod
    def _create_empty_like(arr: np.ndarray, size: Optional[int] = None) -> Optional[np.ndarray]:
        """创建与arr类似但第一维大小为size的空数组，并填充np.nan"""
        if arr is None:
            return None
        # 获取数组第一维大小作为默认值
        size = arr.shape[0] if size is None else size
        # 构建新形状
        new_shape = (size,) if arr.ndim == 1 else (size,) + arr.shape[1:]
        # 创建空数组并填充NaN
        empty_arr = np.empty(new_shape, dtype=arr.dtype)
        empty_arr[:] = np.nan
        # 返回空数组
        return empty_arr

    def __repr__(self) -> str:
        return (f"Solutions(n_sols={self.n_sols}, "
                f"n_vars={self.n_vars}, "
                f"n_objs={self.n_objs}, "
                f"n_cons={self.n_cons})")

    def summary(self) -> str:
        """
        获取简要摘要
        """
        info = [
            f"Solutions object:",
            f"  Number of solutions: {self.n_sols}",
            f"  Number of decision variables: {self.n_vars}",
            f"  Number of objectives: {self.n_objs if self.n_objs is not None else 'Not calculated'}",
            f"  Number of constraints: {self.n_cons if self.n_cons is not None else 'Not calculated'}",
            f"  Fitness values calculated: {self.fits is not None}",
            f"  Metrics: {list(self.metrics.keys())}"
        ]
        return "\n".join(info)

    def __str__(self) -> str:
        """详细字符串表示，显示所有数据内容"""
        lines = list()
        # 设置分隔符数量
        num_sep = 66
        # 基本信息头
        lines.append("=" * num_sep)
        # 维度信息
        lines.append(f"Decision variables: {self.n_sols} × {self.n_vars}")
        lines.append(f"Objectives: {self.n_sols} × {self.n_objs if self.n_objs else 'N/A'}")
        lines.append(f"Constraints:  {self.n_sols} × {self.n_cons if self.n_cons else 'N/A'}")
        lines.append(f"Fitness values: {self.n_sols} × 1" if self.fits is not None else "Fitness values: N/A")
        lines.append("-" * num_sep)
        # 目标值矩阵
        lines.append(self.format_matrix_info("Objectives (objs)", self.f, self.n_objs))
        lines.append(self.format_matrix_info("Constraints (cons)", self.c, self.n_cons))
        lines.append(self.format_matrix_info("Decision Variables (decs)", self.x, self.n_vars))
        lines.append(self.format_matrix_info("Fitness values (fits)", self.t, 1))
        # 指标值
        lines.append("-" * num_sep)
        lines.append("Metrics:")
        if self._metrics:
            for name, value in self._metrics.items():
                lines.append(f"  {name}: {value:.6e}" if isinstance(value, (int, float)) else f"  {name}: {value}")
        else:
            lines.append("  No metrics calculated")
        lines.append("=" * num_sep)

        return "\n".join(lines)

    @staticmethod
    def format_matrix_info(label: str, data, cols: Optional[int] = None, float_format: str = ".6e") -> str:
        """格式化矩阵信息"""
        if data is None:
            return f"{label}: N/A"
        # 将数据转换为numpy数组以便统一处理
        data_arr = np.asarray(data)
        if data_arr.size == 1:  # 单元素数组或标量
            data_value = f"{float(data_arr.flat[0]):{float_format}}"
            return f"{label}: {data_value}"
        data_str = str(data).strip()
        # 判断是否换行
        should_wrap = False
        # 基于列数判断
        if cols and cols > 1:
            should_wrap = True
        # 基于是否已有换行判断
        elif '\n' in data_str:
            should_wrap = True
        if should_wrap:
            return f"{label}:\n{data_str}"
        else:
            return f"{label}: {data_str}"
