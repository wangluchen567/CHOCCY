# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
问题父类
"""

import warnings
import numpy as np
from ..core import ProblemWarning
from ..types import VarType, VarTypeDict
from ..utilities.commons import latin_hypercube
from typing import Union, Dict, List, Optional


class Problem(object):
    # 枚举问题变量的类型
    REAL = VarType.REAL
    INT = VarType.INT
    BIN = VarType.BIN
    PMU = VarType.PMU
    FIX = VarType.FIX
    # 受保护/需提供/可选提供的方法
    PROTECTED_METHODS = {'calc_objs', 'calc_cons'}
    REQUIRED_PAIRS = {('calc_objs_mat', 'calc_obj')}
    OPTIONAL_PAIRS = {
        ('calc_cons_mat', 'calc_con'),  # 约束值
        ('calc_objs_grad_mat', 'calc_obj_grad'),  # 目标梯度
        ('calc_cons_grad_mat', 'calc_con_grad')  # 约束梯度
    }

    def __init__(self,
                 var_types: Union[int, VarType, np.ndarray],
                 n_vars: int,
                 n_objs: int,
                 n_cons: int = 0,
                 l_bounds: Union[float, np.ndarray] = 0.0,
                 u_bounds: Union[float, np.ndarray] = 1.0):
        """
        问题父类

        :param var_types: 问题的变量类型，
                          可以是单个类型或每个维度变量的类型数组
                          (0:实数, 1:整数, 2:二进制, 3:序列, 4:固定标签)
        :param n_vars: 决策变量个数
        :param n_objs: 目标函数个数
        :param n_cons: 约束函数个数
        :param l_bounds: 决策变量下界（包含）
        :param u_bounds: 决策变量上界（实数包含，整数不包含）
        """
        self.n_vars = n_vars  # 决策变量个数
        self.n_objs = n_objs  # 目标函数个数
        self.n_cons = n_cons  # 约束函数个数
        self.var_types = self._format_var_types(var_types)  # 处理问题变量类型
        self.l_bounds, self.u_bounds = self._format_bounds(l_bounds, u_bounds)  # 处理边界输入
        self.unique_types = None  # 问题类型情况(混合情况)
        self.type_to_indices = None  # 每个问题类别对应的位置
        self._get_type_info()  # 获取当前问题存在的类型信息
        self._adjust_bounds()  # 对特定类型的边界进行调整
        self.n_samples = 10000  # 用于指标评估/Pareto前沿绘图 最优解采样数
        # 决策向量表示的标签集，用于固定标签问题
        self.label_set = None
        # 初始化函数映射
        self.init_funcs = VarTypeDict({
            self.REAL: self._init_real_vars,
            self.INT: self._init_integer_vars,
            self.BIN: self._init_binary_vars,
            self.PMU: self._init_permutation_vars,
            self.FIX: self._init_fixed_label_vars
        })

    def __init_subclass__(cls, **kwargs):
        """扩展的子类初始化检查"""
        super().__init_subclass__(**kwargs)
        # 如果类是动态创建的，可以跳过检查
        if getattr(cls, '_skip_checks', False):
            # 仍然需要设置标志，但不做严格检查
            cls._setup_optional_flags()
            return
        # 检查是否错误覆写了受保护的方法
        cls._check_protected_methods()
        # 检查必须至少覆写一个的方法对
        cls._check_required_overrides()
        # 检查重复覆写并发出警告
        cls._check_duplicate_overrides()
        # 设置所有可选功能的标志
        cls._setup_optional_flags()

    def calc_objs(self, xs: np.ndarray) -> np.ndarray:
        """计算目标值（不可覆写）"""
        # 对数据进行浅拷贝，防止其被修改
        xs_ = xs.copy()
        # 确保单个解也是向量表示
        xs_ = self.to_row(xs_)
        # 若为整数问题则需要向下取整
        if self.INT in self.type_to_indices:
            xs_[:, self.type_to_indices[self.INT]] \
                = np.floor(xs_[:, self.type_to_indices[self.INT]])
        objs = self.calc_objs_mat(xs_)
        # 保证二维形状方便并行操作
        return self.to_col(objs)

    def calc_cons(self, xs: np.ndarray) -> np.ndarray:
        """计算约束值（不可覆写）"""
        # 对数据进行浅拷贝，防止其被修改
        xs_ = xs.copy()
        # 确保单个解也是向量表示
        xs_ = self.to_row(xs_)
        # 若为整数问题则需要向下取整
        if self.INT in self.type_to_indices:
            xs_[:, self.type_to_indices[self.INT]] \
                = np.floor(xs_[:, self.type_to_indices[self.INT]])
        cons = self.calc_cons_mat(xs_)
        # 保证二维形状方便并行操作
        return self.to_col(cons)

    def calc_grad(self, xs: np.ndarray) -> Union[np.ndarray, tuple]:
        """
        计算梯度值（不可覆写）
        :param xs: 决策向量
        :return: 若无约束则返回目标函数梯度，否则返回 (目标函数梯度, 约束函数梯度)
        """
        # 对数据进行浅拷贝，防止其被修改
        xs_ = xs.copy()
        # 确保单个解也是向量表示
        xs_ = self.to_row(xs_)
        # 计算目标函数的梯度
        objs_grad = self.calc_objs_grad_mat(xs_)
        # 若约束计算未被覆写则无需计算约束函数梯度
        if not self.has_cons:
            return self.to_col(objs_grad)
        else:  # 否则若约束计算被覆写则需额外计算约束函数梯度
            cons_grad = self.calc_cons_grad_mat(xs_)
            return (self.to_col(objs_grad),
                    self.to_col(cons_grad))

    def calc_objs_mat(self, xs: np.ndarray) -> np.ndarray:
        """给定决策向量矩阵计算目标值矩阵（子类可实现覆写）"""
        # 检查 calc_obj 子类是否已实现覆写
        if self.calc_obj is Problem.calc_obj:
            raise NotImplementedError(
                f"Class '{self.__class__.__name__}' must implement "
                f"at least one of: calc_obj() or calc_objs_mat()"
            )
        n_sols = xs.shape[0]
        objs = np.zeros((n_sols, self.n_objs))
        for i in range(n_sols):
            objs[i] = self.calc_obj(xs[i])
        return objs

    def calc_cons_mat(self, xs: np.ndarray) -> np.ndarray:
        """给定决策向量矩阵计算约束值矩阵（默认无约束）（子类可实现覆写）"""
        n_sols = xs.shape[0]
        # 计算约束的方法没有被覆写 或者约束函数数量为0 则返回空矩阵
        if not self.has_cons or self.n_cons == 0:
            # 无约束问题则返回空矩阵
            return np.empty((n_sols, 0), dtype=float)
        cons = np.zeros((n_sols, self.n_cons))
        for i in range(n_sols):
            cons[i] = self.calc_con(xs[i])
        return cons

    def calc_objs_grad_mat(self, xs: np.ndarray) -> np.ndarray:
        """给定决策向量矩阵计算目标函数的梯度矩阵（子类可实现覆写）"""
        if self.has_objs_grad:
            # 若子类覆写了计算单个解的梯度则使用子类的计算
            n_sols = xs.shape[0]
            objs_grad = np.zeros((n_sols, self.n_vars))
            for i in range(n_sols):
                objs_grad[i] = self.calc_obj_grad(xs[i])
            return objs_grad
        # 否则默认使用有限差分法估计目标函数梯度
        # obj_grad = f(x + epsilon) - f(x) / (x * epsilon)
        xs_ = np.where(xs == 0, 1.e-12, xs)
        # 得到决策变量矩阵的形状
        n_sols, n_vars = xs_.shape
        # 每个解重复 n_vars 次（对应 n_vars 个变量扰动）
        xs_tiled = np.repeat(xs_, n_vars, axis=0)  # (n_sols * n_vars, n_vars)
        # 构造扰动掩码：块对角单位阵
        pert_mask = np.tile(np.eye(n_vars), (n_sols, 1))  # (n_sols * n_vars, n_vars)
        # 施加扰动
        xs_disturb = xs_tiled * (1 + pert_mask * 1e-6)
        # 计算
        objs_base = self.calc_objs(xs_)  # (n_sols, n_objs)
        objs_disturb = self.calc_objs(xs_disturb)  # (n_sols * n_vars, n_objs)
        objs_base_tiled = np.repeat(objs_base, n_vars, axis=0)  # (n_sols * n_vars, n_objs)
        # 差分并 reshape
        diffs = np.asarray(objs_disturb - objs_base_tiled)  # (n_sols * n_vars, n_objs)
        diffs = diffs.reshape(n_sols, n_vars, self.n_objs)  # (n_sols, n_vars, n_objs)
        # 逐元素相除
        objs_grad = diffs / (xs_.reshape(n_sols, n_vars, 1) * 1e-6)
        if self.n_objs == 1:  # 若是多目标则返回完整的多目标雅可比矩阵 (n_sols, n_vars, n_objs)
            objs_grad = objs_grad.reshape(n_sols, n_vars)
        return objs_grad

    def calc_cons_grad_mat(self, xs: np.ndarray) -> np.ndarray:
        """给定决策向量矩阵计算约束函数的梯度矩阵（子类可实现覆写）"""
        if self.has_cons_grad:
            # 若子类覆写了计算单个解的梯度则使用子类的计算
            n_sols = xs.shape[0]
            cons_grad = np.zeros((n_sols, self.n_vars))
            for i in range(n_sols):
                cons_grad[i] = self.calc_con_grad(xs[i])
            return cons_grad
        # 否则默认使用有限差分法估计约束函数梯度
        # con_grad = g(x + epsilon) - g(x) / (x * epsilon)
        xs_ = np.where(xs == 0, 1.e-12, xs)
        n_sols, n_vars = xs_.shape
        # 每个解重复 n_vars 次
        xs_tiled = np.repeat(xs_, n_vars, axis=0)  # (n_sols * n_vars, n_vars)
        # 块对角扰动掩码
        pert_mask = np.tile(np.eye(n_vars), (n_sols, 1))  # (n_sols * n_vars, n_vars)
        # 施加扰动
        xs_disturb = xs_tiled * (1 + pert_mask * 1e-6)
        # 计算
        cons_base = self.calc_cons(xs_)  # (n_sols, n_cons)
        cons_disturb = self.calc_cons(xs_disturb)  # (n_sols * n_vars, n_cons)
        cons_base_tiled = np.repeat(cons_base, n_vars, axis=0)  # (n_sols * n_vars, n_cons)
        # 差分并 reshape
        diffs = np.asarray(cons_disturb - cons_base_tiled)  # (n_sols * n_vars, n_cons)
        diffs = diffs.reshape(n_sols, n_vars, self.n_cons)  # (n_sols, n_vars, n_cons)
        # 逐元素相除
        cons_grad = diffs / (xs_.reshape(n_sols, n_vars, 1) * 1e-6)
        if self.n_cons <= 1:  # 若是多约束则返回完整的多约束雅可比矩阵 (n_sols, n_vars, n_objs)
            cons_grad = cons_grad.reshape(n_sols, n_vars)
        return cons_grad

    def calc_obj(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """计算单个决策向量的目标值（可选实现，但至少实现 calc_obj() 或 calc_objs_mat() 中的一个）"""
        # 检查是否实现了 calc_objs_mat
        if self.calc_objs_mat is Problem.calc_objs_mat:
            raise NotImplementedError(
                f"Problem subclass '{self.__class__.__name__}' must implement "
                f"at least one of: calc_obj() or calc_objs_mat()"
            )
        # 默认使用 calc_objs_mat 计算单个解
        xs = self.to_row(x)
        return self.calc_objs_mat(xs)[0]

    def calc_con(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """
        计算单个决策向量的约束值（可选实现）
        约定：返回值 ≤ 0 表示约束满足
        """
        # 计算约束的方法没有被覆写 或者约束函数数量为0 则返回空数组
        if not self.has_cons or self.n_cons == 0:
            # 无约束问题返回空数组
            return np.array([], dtype=float)
        # 检查是否实现了 calc_cons_mat
        if self.calc_cons_mat is Problem.calc_cons_mat:
            raise NotImplementedError(
                f"Problem subclass '{self.__class__.__name__}' must implement "
                f"at least one of: calc_con() or calc_cons_mat()"
            )
        # 默认使用 calc_cons_mat 计算单个解
        xs = self.to_row(x)
        return self.calc_cons_mat(xs)[0]

    def calc_obj_grad(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """计算单个决策向量的目标函数梯度（可选实现）"""
        # 检查是否实现了 calc_objs_grad_mat
        if self.calc_objs_grad_mat is Problem.calc_objs_grad_mat:
            raise NotImplementedError(
                f"If gradient calculation is required, "
                f"Problem subclass '{self.__class__.__name__}' must implement "
                f"at least one of: calc_obj_grad() or calc_objs_grad_mat()"
            )
        # 默认使用 calc_objs_grad_mat 计算单个解
        xs = self.to_row(x)
        return self.calc_objs_grad_mat(xs)[0]

    def calc_con_grad(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """计算单个决策向量的约束函数梯度（可选实现）"""
        # 检查是否实现了 calc_cons_grad_mat
        if self.calc_cons_grad_mat is Problem.calc_cons_grad_mat:
            raise NotImplementedError(
                f"If gradient calculation is required, "
                f"Problem subclass '{self.__class__.__name__}' must implement "
                f"at least one of: calc_con_grad() or calc_cons_grad_mat()"
            )
        # 默认使用 calc_cons_grad_mat 计算单个解
        xs = self.to_row(x)
        return self.calc_cons_grad_mat(xs)[0]

    @property
    def optimums(self) -> Optional[Union[float, np.ndarray]]:
        """获取理论最优目标值(或参考点向量)（懒加载缓存）"""
        # noinspection PyAttributeOutsideInit
        if not hasattr(self, '_optimums_cached'):
            self._optimums_cached = self.get_optimums()
        return self._optimums_cached

    @property
    def pareto_front(self) -> Optional[Union[list, np.ndarray]]:
        """获取帕累托最优前沿(以绘图)(懒加载缓存）"""
        # noinspection PyAttributeOutsideInit
        if not hasattr(self, '_pareto_front_cached'):
            self._pareto_front_cached = self.get_pareto_front()
        return self._pareto_front_cached

    def get_optimums(self) -> Optional[Union[float, np.ndarray]]:
        """
        获取理论最优目标值(或参考点向量)（子类可覆写）

        此方法被 optimums property 调用并缓存结果，
        子类应覆写此方法而非直接覆写 property。
        """
        return None

    def get_pareto_front(self) -> Optional[Union[list, np.ndarray]]:
        """
        获取帕累托最优前沿(以绘图)（子类可覆写）

        此方法被 pareto_front property 调用并缓存结果，
        子类应覆写此方法而非直接覆写 property。
        """
        return None

    def init_decs_mat(self, n_sols: int, seed: Optional[int] = None) -> np.ndarray:
        """初始化解集的决策向量矩阵（根据问题变量类型随机初始化生成）"""
        # 初始化随机种子
        if seed is not None:
            np.random.seed(seed)
        # 初始化决策向量矩阵
        decs = np.zeros(shape=(n_sols, self.n_vars),
                        dtype=int if np.all(VarType.convert(self.unique_types) > 2) else float)
        # 若需要随机初始化生成的个数为0 则直接返回
        if n_sols == 0:
            return decs
        # 按类型初始化各个部分
        for var_type in self.unique_types:
            indices = self.type_to_indices[var_type]  # 该类型的变量索引
            if len(indices) > 0:  # 确保有这种类型的变量
                var_values = self.init_funcs[var_type](n_sols, indices)
                decs[:, indices] = var_values
        return decs

    def _init_real_vars(self, n_sols: int, indices: np.ndarray) -> np.ndarray:
        """初始化求解实数问题的决策向量矩阵"""
        # 使用 拉丁超立方采样 初始化决策向量矩阵
        n_vars, l_bounds, u_bounds = len(indices), self.l_bounds[indices], self.u_bounds[indices]
        return latin_hypercube(l_bounds, u_bounds, size=(n_sols, n_vars))

    def _init_integer_vars(self, n_sols: int, indices: np.ndarray) -> np.ndarray:
        """初始化求解整数问题的决策向量矩阵"""
        # 使用 拉丁超立方采样 初始化决策向量矩阵
        n_vars, l_bounds, u_bounds = len(indices), self.l_bounds[indices], self.u_bounds[indices]
        return latin_hypercube(l_bounds, u_bounds, size=(n_sols, n_vars))

    @staticmethod
    def _init_binary_vars(n_sols: int, indices: np.ndarray) -> np.ndarray:
        """初始化求解二进制问题的决策向量矩阵"""
        return np.asarray(np.random.randint(2, size=(n_sols, len(indices))))

    @staticmethod
    def _init_permutation_vars(n_sols: int, indices: np.ndarray) -> np.ndarray:
        """初始化求解序列问题的决策向量矩阵"""
        decs = np.argsort(np.random.uniform(0, 1, size=(n_sols, len(indices))), axis=1)
        # 为保证纯启发式算法(如LocalSearch相关)优化时不存在随机性，将第一个决策向量设置为纯正序
        decs[0, :] = np.arange(len(indices))
        return decs

    def _init_fixed_label_vars(self, n_sols: int, indices: np.ndarray) -> np.ndarray:
        """初始化求解固定标签问题的决策向量矩阵"""
        n_labels = len(indices)
        # 获取标签集合
        if hasattr(self, 'label_set') and self.label_set is not None:
            if len(self.label_set) != n_labels:
                raise ValueError(
                    f"label_set size ({len(self.label_set)}) "
                    f"must match number of positions ({n_labels})"
                )
            labels = np.asarray(self.label_set)
        else:
            # 默认使用 0 到 n_labels-1 作为标签
            labels = np.arange(n_labels)
        # 为每个解生成一个随机排列
        decs = np.zeros((n_sols, n_labels), dtype=labels.dtype)
        for i in range(n_sols):
            decs[i] = np.random.permutation(labels)
        return decs

    @staticmethod
    def to_row(x: np.ndarray) -> np.ndarray:
        """将输入转换为行向量（形状为 (1, n)）"""
        x = np.asarray(x)
        return x.reshape(1, -1) if x.ndim <= 1 else x

    @staticmethod
    def to_col(x: np.ndarray) -> np.ndarray:
        """将输入转换为列向量（形状为 (n, 1)）"""
        x = np.asarray(x)
        return x.reshape(-1, 1) if x.ndim <= 1 else x

    def _format_var_types(self, var_types: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """格式化问题类型数组"""
        if isinstance(var_types, (int, np.integer)):
            # 单个类型应用到所有维度
            result = np.full(self.n_vars, int(var_types), dtype=int)
        elif isinstance(var_types, (list, np.ndarray)):
            # 数组类型，检查长度
            result = np.asarray(var_types, dtype=int)
            if len(result) != self.n_vars:
                raise ValueError(
                    f"var_types length {len(result)} must match n_vars {self.n_vars}"
                )
        else:
            raise TypeError(
                f"var_types must be int, list or np.ndarray, got {type(var_types)}"
            )
        # 验证类型值是否有效
        valid_types = {t.value for t in VarType}
        # 获取实际变量类型
        invalid_types = set(VarType.convert(result)) - valid_types
        if invalid_types:
            raise ValueError(
                f"Invalid variable types found: {invalid_types}. "
                f"Valid types are: {sorted(valid_types)}"
            )
        return result

    def _format_bounds(self, l_bounds, u_bounds):
        """格式化并验证边界"""
        # 格式化下界
        lbs_array = self.format_to_arr(l_bounds, "l_bounds")
        ubs_array = self.format_to_arr(u_bounds, "u_bounds")
        # 验证边界维度
        if lbs_array.shape != (self.n_vars,):
            raise ValueError(f"l_bounds must have shape ({self.n_vars},), got {lbs_array.shape}")
        if ubs_array.shape != (self.n_vars,):
            raise ValueError(f"u_bounds must have shape ({self.n_vars},), got {ubs_array.shape}")
        # 验证边界有效性
        if not np.all(lbs_array <= ubs_array):
            invalid_indices = np.where(lbs_array > ubs_array)[0]
            raise ValueError(
                f"l_bounds must be <= u_bounds for all dimensions. "
                f"Violations at indices: {invalid_indices}"
            )
        return lbs_array.astype(float), ubs_array.astype(float)

    def format_to_arr(self, value, name: str) -> np.ndarray:
        """将给定值格式化为与问题维度相同的数组"""
        # 处理标量
        if isinstance(value, (int, float)):
            return np.full(self.n_vars, float(value))
        # 处理可迭代对象
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).flatten()  # flatten确保一维
            if len(arr) != self.n_vars:
                raise ValueError(
                    f"{name} length ({len(arr)}) must equal problem dimension ({self.n_vars})"
                )
            return arr
        # 不支持的类型
        raise TypeError(
            f"{name} must be scalar, list, tuple or numpy array, "
            f"got {type(value).__name__}"
        )

    def _get_type_info(self):
        """分析类型信息"""
        # 获取唯一类型
        self.unique_types = np.unique(self.var_types)
        # 构建类型到索引的映射
        self.type_to_indices: Dict[int, np.ndarray] = {}
        for t in self.unique_types:
            self.type_to_indices[t] = np.where(self.var_types == t)[0]

    def _adjust_bounds(self):
        """应用类型特定的边界调整"""
        # 整数类型：上界不包含
        if self.INT in self.type_to_indices:
            int_indices = self.type_to_indices[self.INT]
            epsilon = 1e-9  # 使用命名常量
            self.u_bounds[int_indices] -= epsilon
        # 二进制类型：强制设置为[0, 1]
        if self.BIN in self.type_to_indices:
            bin_indices = self.type_to_indices[self.BIN]
            self.l_bounds[bin_indices] = 0.0
            self.u_bounds[bin_indices] = 1.0

    @classmethod
    def _check_protected_methods(cls):
        """检查是否错误覆写了受保护的方法"""
        for method_name in cls.PROTECTED_METHODS:
            if getattr(cls, method_name) != getattr(Problem, method_name):
                raise TypeError(
                    f"Class '{cls.__name__}' cannot override protected method '{method_name}'.\n"
                    f"Please override the corresponding '_mat' and '_single' methods instead.\n"
                    f"For '{method_name}', you should override: "
                    f"'{method_name}_mat' and/or '{method_name.rstrip('s')}'"
                )

    @classmethod
    def _check_required_overrides(cls):
        """检查必须至少覆写一个的方法对"""
        for mat_method, single_method in cls.REQUIRED_PAIRS:
            mat_overridden = getattr(cls, mat_method) != getattr(Problem, mat_method)
            single_overridden = getattr(cls, single_method) != getattr(Problem, single_method)

            if not (mat_overridden or single_overridden):
                raise TypeError(
                    f"Class '{cls.__name__}' must implement at least one calculation method.\n"
                    f"Please override either '{mat_method}' (for batch optimization) "
                    f"or '{single_method}' (for single solution)."
                )

    @classmethod
    def _check_duplicate_overrides(cls):
        """检查重复覆写并发出警告"""
        for mat_method, single_method in cls.REQUIRED_PAIRS:
            mat_overridden = getattr(cls, mat_method) != getattr(Problem, mat_method)
            single_overridden = getattr(cls, single_method) != getattr(Problem, single_method)

            if mat_overridden and single_overridden:
                warnings.warn(
                    f"Class '{cls.__name__}' overrides both '{mat_method}' and '{single_method}'.\n"
                    f"The '{mat_method}' method will be used for calculations.\n"
                    f"Consider removing the '{single_method}' override to simplify your implementation.",
                    ProblemWarning, stacklevel=2
                )

    @classmethod
    def _setup_optional_flags(cls):
        """设置所有可选功能的标志"""
        # 约束值
        cons_mat_overridden = getattr(cls, 'calc_cons_mat') != getattr(Problem, 'calc_cons_mat')
        con_overridden = getattr(cls, 'calc_con') != getattr(Problem, 'calc_con')
        cls._has_cons = cons_mat_overridden or con_overridden
        # 目标梯度
        objs_grad_mat_overridden = getattr(cls, 'calc_objs_grad_mat') != getattr(Problem, 'calc_objs_grad_mat')
        obj_grad_overridden = getattr(cls, 'calc_obj_grad') != getattr(Problem, 'calc_obj_grad')
        cls._has_objs_grad = objs_grad_mat_overridden or obj_grad_overridden
        # 约束梯度
        cons_grad_mat_overridden = getattr(cls, 'calc_cons_grad_mat') != getattr(Problem, 'calc_cons_grad_mat')
        con_grad_overridden = getattr(cls, 'calc_con_grad') != getattr(Problem, 'calc_con_grad')
        cls._has_cons_grad = cons_grad_mat_overridden or con_grad_overridden

    @property
    def has_cons(self):
        """是否定义了自定义约束（缓存到实例属性）"""
        cache_attr = '_has_cons_cached'
        if not hasattr(self, cache_attr):
            # 实例没有自己的缓存，从类获取默认值
            setattr(self, cache_attr, self._has_cons)
        return getattr(self, cache_attr)

    @property
    def has_objs_grad(self):
        """是否提供了目标函数梯度（缓存到实例属性）"""
        cache_attr = '_has_objs_grad_cached'
        if not hasattr(self, cache_attr):
            # 实例没有自己的缓存，从类获取默认值
            setattr(self, cache_attr, self._has_objs_grad)
        return getattr(self, cache_attr)

    @property
    def has_cons_grad(self):
        """是否提供了约束函数梯度（缓存到实例属性）"""
        cache_attr = '_has_cons_grad_cached'
        if not hasattr(self, cache_attr):
            # 实例没有自己的缓存，从类获取默认值
            setattr(self, cache_attr, self._has_cons_grad)
        return getattr(self, cache_attr)

    def plot_by_problem(self, n_iter: Optional[int] = None, best=None, **kwargs):
        """问题提供的绘图函数（可由子类覆写实现）"""
        pass

    def get_info(self):
        """获取问题的相关信息"""
        return {
            'var_types': self.var_types.tolist(),
            'n_vars': self.n_vars,
            'n_objs': self.n_objs,
            'n_cons': self.n_cons,
            'l_bounds': self.l_bounds.tolist(),
            'u_bounds': self.u_bounds.tolist(),
            'unique_types': self.unique_types.tolist(),
        }


def _make_method(func):
    """将普通函数转换为实例方法"""
    if func is None:
        return None
    if hasattr(func, '__self__'):  # 已经是方法
        return func
    # 创建实例方法
    return lambda self, *args, **kwargs: func(*args, **kwargs)


def create_problem(
        calc_objs_mat=None,
        calc_cons_mat=None,
        calc_objs_grad_mat=None,
        calc_cons_grad_mat=None,
        calc_obj=None,
        calc_con=None,
        calc_obj_grad=None,
        calc_con_grad=None,
        var_types: Union[int, VarType, np.ndarray] = Problem.REAL,
        n_vars: int = 1,
        n_objs: int = 1,
        n_cons: int = 0,
        l_bounds: Union[float, np.ndarray] = 0.0,
        u_bounds: Union[float, np.ndarray] = 1.0,
        name: str = "DynamicProblem",
        **kwargs
):
    """
    动态创建问题子类并返回实例
    使用示例：
    >>> def my_objective(x):
    ...     return np.sum(x**2)
    >>>
    >>> problem = create_problem(
    ...     calc_obj=my_objective,
    ...     n_vars=2,
    ...     n_objs=1,
    ...     var_types=Problem.REAL,
    ...     l_bounds=-10,
    ...     u_bounds=10
    ... )
    注：calc_objs_mat 和 calc_obj 至少需提供其中一个
    :param calc_objs_mat: 批量目标函数（接收决策矩阵，返回目标值矩阵）
    :param calc_cons_mat: 批量约束函数（接收决策矩阵，返回约束值矩阵）
    :param calc_objs_grad_mat: 批量目标梯度函数（接收决策矩阵，返回梯度矩阵）
    :param calc_cons_grad_mat: 批量约束梯度函数（接收决策矩阵，返回梯度矩阵）
    :param calc_obj: 单点目标函数（接收单个决策向量，返回目标值），与 calc_objs_mat 二选一
    :param calc_con: 单点约束函数（接收单个决策向量，返回约束值），与 calc_cons_mat 二选一
    :param calc_obj_grad: 单点目标梯度函数，与 calc_objs_grad_mat 二选一
    :param calc_con_grad: 单点约束梯度函数，与 calc_cons_grad_mat 二选一
    :param var_types: 决策变量类型，默认实数
    :param n_vars: 决策变量个数
    :param n_objs: 目标个数
    :param n_cons: 约束个数
    :param l_bounds: 决策变量下界
    :param u_bounds: 决策变量上界
    :param name: 动态生成的问题类名
    :param kwargs: 额外属性，将作为类属性注入问题子类
    :return: 初始化后的问题对象
    """

    # 创建动态子类，并设置跳过检查的标志
    class DynamicProblem(Problem):
        _skip_checks = True

    # 设置类名
    DynamicProblem.__name__ = name
    DynamicProblem.__qualname__ = name

    # 将所有函数转换为方法
    methods = {
        'calc_objs_mat': calc_objs_mat,
        'calc_cons_mat': calc_cons_mat,
        'calc_objs_grad_mat': calc_objs_grad_mat,
        'calc_cons_grad_mat': calc_cons_grad_mat,
        'calc_obj': calc_obj,
        'calc_con': calc_con,
        'calc_obj_grad': calc_obj_grad,
        'calc_con_grad': calc_con_grad,
    }

    for method_name, func in methods.items():
        if func is not None:
            setattr(DynamicProblem, method_name, _make_method(func))

    # 处理额外的类属性
    for key, value in kwargs.items():
        if not key.startswith('_'):
            setattr(DynamicProblem, key, value)

    # 验证至少有一个目标函数计算方法
    if calc_objs_mat is None and calc_obj is None:
        raise ValueError(
            f"Must provide at least one objective calculation method: "
            f"calc_objs_mat or calc_obj"
        )

    # 返回实例
    return DynamicProblem(
        var_types=var_types,
        n_vars=n_vars,
        n_objs=n_objs,
        n_cons=n_cons,
        l_bounds=l_bounds,
        u_bounds=u_bounds
    )
