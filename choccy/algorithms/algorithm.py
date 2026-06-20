# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
算法父类
"""

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Union, Optional

from .. import __version__
from ..problems import Problem
from ..solutions import Solutions
from ..utilities.metrics import *
from ..utilities.strategies import *
from ..utilities.visualization import *
from ..utilities.logging import setup_logger
from ..utilities.commons import calc_penalized_objs
from ..types import VarType, VisualMode, MetricType, VarTypeDict
from ..core import record_time, AlgorithmError, VisualizationError


class Algorithm(object):
    # 枚举问题变量的类型
    REAL = VarType.REAL
    INT = VarType.INT
    BIN = VarType.BIN
    PMU = VarType.PMU
    FIX = VarType.FIX

    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 visual_mode: Optional[str] = None):
        """
        算法父类

        Code Maintainer: LuChen Wang
        :param n_sols: 解集大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param visual_mode: 可视化模式
        """
        # 初始化给定参数
        self.n_sols = n_sols
        self.max_iter = max_iter
        # 初始化交叉、变异概率
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        # 初始化问题实例
        self.problem = None
        # 算法配置：是否仅支持单目标
        self.single_obj_only = None
        # 算法配置：支持的决策变量类型
        self.supported_var_types = None
        # 初始化解集与最优解
        self.sols = Solutions(decs=np.empty(0))
        self.best = None
        # 初始化解集与最优解的历史记录
        self.history_sols = []
        self.history_best = []
        # 记录迭代次数
        self.n_iter = 0
        # 记录运行时间
        self.run_time = 0.0
        # 初始化追踪的指标
        self._metrics = None
        # 保存追踪指标集合(格式化后)
        self._metrics_set = []
        # 初始化是否请求追踪的标记
        self._tracking_requested = False
        # 初始化日志记录器
        self.logger = None
        # 初始化可视化模式(默认进度条显示)
        self.visual_mode = 'progress' \
            if visual_mode is None else visual_mode
        # 初始化默认可视化配置
        self.visual_config = {
            'fig_size': (6.4, 4.8),
            'interval': 30,
            'metric': None,
            'pareto_front': None,
            'save_path': None,
        }
        # 初始化动画绘制器
        self.animator = Animator()
        # 初始化遗传进化操作算子函数映射
        self.operator_funcs = VarTypeDict({
            self.REAL: operator_real,
            self.INT: operator_real,
            self.BIN: operator_binary,
            self.PMU: operator_permutation,
            self.FIX: operator_fix_label,
        })
        # 初始化差分进化操作算子函数映射
        self.operator_de_funcs = VarTypeDict({
            self.REAL: operator_de_real,
            self.INT: operator_de_real,
            self.BIN: operator_de_binary,
        })
        # 初始化粒子群优化操作算子函数映射
        self.operator_pso_funcs = VarTypeDict({
            self.REAL: operator_pso_real,
            self.INT: operator_pso_real,
            self.BIN: operator_pso_binary,
        })
        # 初始化绘图函数映射
        self.plot_funcs = {
            VisualMode.METRICS: self.plot_metrics,
            VisualMode.DECISIONS: self.plot_decisions,
            VisualMode.OBJECTIVES: self.plot_objectives,
            VisualMode.HYBRIDS_2D: self.plot_hybrids_2d,
            VisualMode.HYBRIDS_3D: self.plot_hybrids_3d,
            VisualMode.CUSTOM_PROBLEM: self.plot_by_problem,
            VisualMode.CUSTOM_ALGORITHM: self.plot_by_algorithm
        }
        # 初始化评估函数映射
        self.eval_metric_funcs = {
            MetricType.HV: self.eval_hv,
            MetricType.GD: self.eval_gd,
            MetricType.IGD: self.eval_igd,
            MetricType.GD_PLUS: self.eval_gd_plus,
            MetricType.IGD_PLUS: self.eval_igd_plus,
            MetricType.PENALIZED_OBJ: self.eval_penalized_obj,
        }

    def __repr__(self):
        """方便查看实例信息"""
        attrs = []
        # 只显示非私有属性
        for key in sorted(self.__dict__.keys()):
            if not key.startswith('_'):
                value = getattr(self, key)
                attrs.append(f"{key}={value!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def _init_algorithm(self, problem: Problem):
        """
        初始化算法
        :param problem: 问题实例
        """
        # 设置问题实例
        self.problem = problem
        # 初始化算法所有参数
        self.init_parameters()
        # 验证算法与给定问题的兼容性
        self._verify_compatibility()

    def init_parameters(self):
        """初始化算法所有参数"""
        # 初始化算法默认参数
        self.cross_prob = 1.0 if self.cross_prob is None else self.cross_prob
        # 算法的变异率由问题的维度确定
        self.mutate_prob = 1 / self.problem.n_vars if self.mutate_prob is None else self.mutate_prob
        # 初始化算法默认配置
        # 默认多目标与单目标 全部支持
        self.single_obj_only = False if self.single_obj_only is None else self.single_obj_only
        # 默认支持全部变量类型的问题
        self.supported_var_types = list(VarType.__members__.values()) \
            if self.supported_var_types is None else self.supported_var_types
        # 对可视化模式进行解析
        self.visual_mode = VisualMode.parse(self.visual_mode)
        # 获取帕累托前沿数据，并更新可视化配置
        self.visual_config['pareto_front'] = self.problem.pareto_front
        # 设置日志记录器
        self.logger = setup_logger() if self.logger is None else self.logger

    def _verify_compatibility(self):
        """验证算法与给定问题的兼容性"""
        # 检查：目标数量兼容性
        if self.single_obj_only and self.problem.n_objs > 1:
            raise AlgorithmError(
                f"Algorithm '{self.__class__.__name__}' is for single-objective problems only. "
                f"Given problem has {self.problem.n_objs} objectives."
            )
        # 检查：变量类型兼容性
        actual_types = VarType.convert(self.problem.unique_types)  # 转换为实际的变量类型
        if not np.all(np.isin(actual_types, self.supported_var_types)):  # 检查是否支持指定的变量类型
            unsupported_types = actual_types[~np.isin(actual_types, self.supported_var_types)]
            raise AlgorithmError(
                f"Algorithm '{self.__class__.__name__}' supports {self.supported_var_types} variable types only. "
                f"Given problem has unsupported types: {unsupported_types.tolist()}."
            )

    @record_time
    def _init_solutions(self, *args, **kwargs):
        """初始化解集（用于记录时间信息）"""
        self.init_solutions(*args, **kwargs)

    def init_solutions(self,
                       seeds: Optional[np.ndarray] = None,
                       shuffle: bool = False):
        """
        初始化解集（支持先验种子解集）
        :param seeds: 先验 种子 解集
        :param shuffle: 是否打乱解集
        """
        # 初始化解集
        self.sols = Solutions(np.empty((0, self.problem.n_vars)))
        # 创建先验种子解集
        n_seeds = 0
        if seeds is not None:
            seeds_sols = Solutions(seeds)
            # 统计先验种子解的个数（按算法设置解个数截断）
            n_seeds = min(self.n_sols, len(seeds_sols))
            self.sols = self.sols.concat(seeds_sols[:n_seeds])
        # 计算需要初始化的随机解个数
        n_random = self.n_sols - n_seeds
        if n_random > 0:
            # 初始化随机解的集合
            random_sols = Solutions(self.problem.init_decs_mat(n_random))
            self.sols = self.sols.concat(random_sols)
        if shuffle:
            # 打乱解集
            self.sols.shuffle(inplace=True)
        # 设置解集的评估函数
        self.set_evaluate_funcs()
        # 对初始解集进行评估并更新最优解
        self.evaluate_and_update()

    def set_evaluate_funcs(self):
        """设置解集的评估函数"""
        # 设置解集的评估函数（目标函数/约束函数/适应度函数）
        self.sols.set_eval_funcs(objs_func=self.eval_objs,
                                 cons_func=self.eval_cons,
                                 fits_func=self.eval_fits)
        # 设置需要追踪计算的性能指标
        self.set_tracked_metrics()

    def evaluate_and_update(self):
        """对解集进行评估并更新最优解"""
        # 对解集进行评估（基础信息）
        self.sols.evaluate()
        # 更新解集的最优解信息
        self.update_best()

    def track_metrics(self, metrics: Optional[Union[str, list, MetricType]] = None):
        """
        :param metrics: 设置要追踪的性能指标
        """
        self._metrics = metrics  # 存储指标配置
        self._tracking_requested = True  # 追踪已请求

    def set_tracked_metrics(self):
        """设置需要追踪计算的性能指标"""
        if not self._tracking_requested:
            # 若未请求追踪则不进行设置
            return
        # 确定要设置的指标枚举列表
        metrics_set = []
        if self._metrics is None:
            # 设置默认指标
            if self.problem.n_objs == 1:
                metrics_set.append(MetricType.PENALIZED_OBJ)
            else:
                metrics_set.append(MetricType.HV)
        else:
            # 统一转换为列表进行处理
            if isinstance(self._metrics, (str, MetricType)):
                # 单个指标，放入列表以便统一循环
                items = [self._metrics]
            else:
                # 已经是列表或其他可迭代对象
                items = self._metrics
            # 设置多个追踪的性能指标
            for item in items:
                metrics_set.append(MetricType.parse(item))
        # 统一设置所有指标函数
        for metric in metrics_set:
            if metric:
                # 确保指标集合无重复
                if metric.value in self._metrics_set:
                    continue
                # 保存格式化后的指标到记录中（保存为str）
                self._metrics_set.append(metric.value)
                # 添加给定的指标（初始化为nan）
                self.sols.add_metric(metric.value, float('nan'))
                # 添加指标评估函数
                self.sols.set_metric_func(
                    metric.value, self.eval_metric_funcs[metric]
                )

    def optimize(self,
                 problem: Problem,
                 seeds: Optional[np.ndarray] = None,
                 shuffle: bool = False):
        """
        主函数入口
        :param problem: 问题实例
        :param seeds: 先验 种子 解集
        :param shuffle: 是否打乱解集
        """
        # 初始化算法相关参数
        self._init_algorithm(problem)
        # 初始化解集(支持先验解集并打乱)
        self._init_solutions(seeds, shuffle)
        # 运行算法求解问题
        self.run()

    def record_state(self):
        """记录当前解集状态"""
        # 记录解集到历史记录中
        self.history_sols.append(self.sols.copy())
        # 记录最优解到历史记录中
        self.history_best.append(self.best.copy())

    def iterator(self):
        """构建迭代器"""
        if self.visual_mode == VisualMode.PROGRESS:
            return tqdm(range(1, self.max_iter + 1))
        else:
            return range(1, self.max_iter + 1)

    def run(self):
        """运行算法"""
        # 迭代次数置零
        self.n_iter = 0
        # 运行前准备工作
        self._prepare()
        # 记录当前解集状态
        self.record_state()
        # 绘制初始状态图
        self.plot(n_iter=self.n_iter, static=False)
        # 算法迭代并优化问题
        for self.n_iter in self.iterator():
            # 运行单步算法
            self._run_step(self.n_iter)
            # 记录解集状态
            self.record_state()
            # 绘制迭代过程中每步状态
            self.plot(n_iter=self.n_iter, static=False)

    @record_time
    def _prepare(self):
        """迭代运行前的准备工作（种群已初始化）（不可覆写）"""
        self.prepare()

    def prepare(self):
        """迭代运行前的准备工作（种群已初始化）（子类可覆写）"""
        pass

    @record_time
    def _run_step(self, iteration: int):
        """运行算法单步（不可覆写）"""
        self.run_step(iteration)

    def run_step(self, iteration: int):
        """运行算法单步（由子类覆写）"""
        pass

    def eval_objs(self, sols: Solutions) -> np.ndarray:
        """评估计算解集的目标值矩阵"""
        return self.problem.calc_objs(sols.xs)

    def eval_cons(self, sols: Solutions) -> np.ndarray:
        """评估计算解集的约束值矩阵"""
        return self.problem.calc_cons(sols.xs)

    def eval_grad(self, sols: Solutions) -> np.ndarray:
        """评估计算解集的梯度值矩阵"""
        return self.problem.calc_grad(sols.xs)

    def eval_fits(self, sols: Solutions) -> np.ndarray:
        """评估计算解集的适应度值向量(默认单目标)"""
        if self.problem.n_objs > 1:
            # 若是多目标则默认返回0
            # 需要每种算法单独覆写
            return np.zeros(len(sols))
        # 单目标返回经过约束惩罚处理后的目标值
        return self.eval_penalized_objs(sols).flatten()

    def eval_hv(self, sols: Solutions) -> float:
        """评估解集的 HV 指标"""
        return calc_hv(objs=np.asarray(sols.get_best().objs), optimums=self.problem.optimums)

    def eval_gd(self, sols: Solutions) -> float:
        """评估解集的 GD 指标"""
        return calc_gd(objs=np.asarray(sols.get_best().objs), optimums=self.problem.optimums)

    def eval_igd(self, sols: Solutions) -> float:
        """评估解集的 IGD 指标"""
        return calc_igd(objs=np.asarray(sols.get_best().objs), optimums=self.problem.optimums)

    def eval_gd_plus(self, sols: Solutions) -> float:
        """评估解集的 GD+ 指标"""
        return calc_gd_plus(objs=np.asarray(sols.get_best().objs), optimums=self.problem.optimums)

    def eval_igd_plus(self, sols: Solutions) -> float:
        """评估解集的 IGD+ 指标"""
        return calc_igd_plus(objs=np.asarray(sols.get_best().objs), optimums=self.problem.optimums)

    def eval_penalized_obj(self, sols: Solutions) -> float:
        """评估解集的约束惩罚后的最优目标值"""
        # 返回约束惩罚后的最优目标值（作为分数）
        return calc_penalized_objs(np.asarray(sols.objs), sols.cons).flat[0]

    def eval_penalized_objs(self, sols: Solutions) -> np.ndarray:
        """评估计算约束惩罚后的目标值矩阵"""
        # 返回约束惩罚后的最优目标值矩阵（作为带约束的目标值矩阵）
        return calc_penalized_objs(np.asarray(sols.objs), sols.cons)

    def apply_operator(self, mating_indices: np.ndarray) -> Solutions:
        """
        执行算子操作进行生成新一代解集
        :param mating_indices: 配对池索引
        :return: 新一代解集（必须由原解集生成）
        """
        # 根据配对池索引创建新解集
        new_sols = self.sols[mating_indices]
        # 按类型操作各个部分
        for var_type in self.problem.unique_types:
            indices = self.problem.type_to_indices[var_type]  # 该类型的变量索引
            # if len(indices) > 0:  # 确保有这种类型的变量
            new_sols.xs[:, indices] \
                = self.operator_funcs[var_type](new_sols.xs[:, indices],
                                                self.problem.l_bounds[indices],
                                                self.problem.u_bounds[indices],
                                                self.cross_prob, self.mutate_prob)
        return new_sols

    def apply_education(self, *args, **kwargs):
        """对新一代解集进行教育(等价于局部搜索)"""
        pass

    def get_mating_indices(self,
                           next_size: Optional[int] = None,
                           p: Union[int, bool] = 2) -> np.ndarray:
        """
        配对池选择（生成配对下标）
        :param next_size: 进入下一代解集的解数量
        :param p: 额外参数, 锦标赛参数k/轮盘选择是否可重复选
        :return: 配对池（下标）
        """
        # 设置默认下一代解集的解数量
        next_size = self.n_sols if next_size is None else next_size
        if isinstance(p, int) and p >= 2:
            # 使用锦标赛选择法获取配对池
            mating_indices = select_by_tournament(np.asarray(self.sols.fits), next_size, p)
        elif isinstance(p, bool):
            # 使用轮盘选择法获取配对池
            mating_indices = select_by_roulette(np.asarray(self.sols.fits), next_size, p)
        else:
            raise AlgorithmError(f"The parameter setting error: {p}")
        return mating_indices

    def apply_selection(self, next_size: int):
        """
        使用选择策略选择进入下一代解集的解索引
        :param next_size: 进入下一代解集的解数量
        :return: 进入下一代解集的解索引
        """
        # 默认根据适应度使用精英选择策略进行选择
        return select_by_elitism(np.asarray(self.sols.fits), next_size)

    def global_selection(self, new_sols: Solutions):
        """
        将原始解集与新一代解集合并后进行全局竞争选择
        :param new_sols: 新一代解集（必须由原解集生成）
        """
        # 计算新一代解集的目标值与约束值
        new_sols.eval_objs().eval_cons()
        # 将原始解集与新一代解集合并
        self.sols = self.sols.concat(new_sols, ignore_warn=True)
        # 对合并后的解集重新进行适应度评估以选择下一代
        self.sols.eval_fits()
        # 使用选择策略选择进入下一代解集的解索引
        better = self.apply_selection(self.n_sols)
        # 全局竞争后的优胜者进入下一代解集
        self.sols = self.sols[better]
        # 更新下一代解集的最优解信息
        self.update_best()
        # 返回选择的索引情况
        return better

    def local_selection(self, new_sols: Solutions):
        """
        原始解集与新一代解集不合并而进行一对一的局部竞争选择
        :param new_sols: 新一代解集（必须由原解集生成）
        """
        # 计算新一代解集的目标值与约束值
        new_sols.eval_objs().eval_cons()
        # 将原始解集与新一代解集合并
        self.sols = self.sols.concat(new_sols, ignore_warn=True)
        # 对合并后的解集重新进行适应度评估以选择下一代
        self.sols.eval_fits()
        # 一对一进行局部竞争选择（前一半是原种群，后一半是新一代种群）
        half_better = self.sols.fits[:self.n_sols] <= self.sols.fits[self.n_sols:]
        # 若新种群更优则使用新种群个体（一对一竞争选择）
        better = np.where(
            half_better,
            np.arange(self.n_sols),
            np.arange(self.n_sols, 2 * self.n_sols)
        )
        # 局部竞争后的优胜者进入下一代解集
        self.sols = self.sols[better]
        # 更新下一代解集的最优解信息
        self.update_best()
        # 返回选择的索引情况
        return better

    def environmental_selection(self, new_sols: Solutions):
        """
        进行环境选择（默认使用全局竞争）
        :param new_sols: 下一代解集
        """
        # 通过全局竞争选择得到下一代解集
        self.global_selection(new_sols)

    def update_best(self):
        """更新最优解(默认使用当前最优解)"""
        self.update_best_current()

    def update_best_current(self):
        """根据当前规则更新最优解"""
        self.best = self.sols.get_best()
        # 在全种群上评估指标（可能包含需要全种群数据的多样性指标）
        self.sols.eval_metrics()
        # 最优解的指标同步全种群的指标
        self.best.metrics = self.sols.metrics.copy()

    def update_best_global(self):
        """根据全局规则更新最优解"""
        if self.best is None:
            self.best = self.sols.get_best()
        else:
            self.best = self.best.concat(self.sols.get_best()).get_best()
        # 对最优解的指标进行评估
        self.best.eval_metrics()

    def finalize(self, sols: Solutions, inplace: bool = False) -> Solutions:
        """
        对解进行输出前的最终处理（如整数变量取整）
        所有解在输出前都应调用此方法，以确保符合问题定义的变量类型要求
        :param sols: 给定解集
        :param inplace: 是否替换
        :return: 处理后的解集
        """
        sols_ = sols if inplace else sols.copy()
        # 处理整数变量
        if self.problem.INT in self.problem.unique_types:
            int_idx = self.problem.type_to_indices[self.problem.INT]
            sols_.xs[:, int_idx] = np.floor(sols_.xs[:, int_idx])
        # # 处理二进制变量
        # if self.problem.BIN in self.problem.unique_types:
        #     bin_idx = self.problem.type_to_indices[self.problem.BIN]
        #     sols_.xs[:, bin_idx] = np.clip(np.round(sols_.xs[:, bin_idx]), 0, 1)
        return sols_

    @property
    def history_xs(self):
        """获取解集的历史决策变量矩阵记录"""
        return self.history_decs

    @property
    def history_decs(self):
        """获取解集的历史决策变量矩阵记录"""
        return [sols.decs for sols in self.history_sols]

    @property
    def history_objs(self):
        """获取解集的历史目标矩阵记录"""
        return [sols.objs for sols in self.history_sols]

    @property
    def history_cons(self):
        """获取解集的历史约束值记录"""
        return [sols.cons for sols in self.history_sols]

    def set_visual_config(self, **kwargs):
        """更新绘图配置"""
        self.visual_config.update(kwargs)
        # 若与动画器参数相关则重新配置动画器参数
        self.animator.set_figsize(self.visual_config.get('fig_size', (6.4, 4.8)))
        self.animator.set_interval(self.visual_config.get('interval', 30))
        self.animator.set_save_frames(self.visual_config.get('save_frames', True))

    def plot(self,
             visual_mode: Optional[str] = None,
             n_iter: Optional[int] = None,
             static: bool = True) -> Optional[Frame]:
        """
        可视化函数，根据不同模式进行可视化绘图
        :param visual_mode: 可视化模式，若不指定则使用默认
        :param n_iter: 当前迭代次数，显示在标题中
        :param static: 是否静态模式显示
        """
        frame = None
        # 可视化模式设置
        if visual_mode is None:
            # 若无指定则使用当前默认模式
            visual_mode = self.visual_mode
        else:
            # 若指定可视化模式则重新解析
            visual_mode = VisualMode.parse(visual_mode)
        # 根据可视化模式绘制或记录数据
        if (visual_mode == VisualMode.NONE or
                visual_mode is VisualMode.PROGRESS):
            # 若不显示或显示进度条则无需绘图
            pass
        elif visual_mode == VisualMode.LOG:
            # 若是日志模式则进行日志输出
            self.logger.info(self.format_log_line())
        elif visual_mode in self.plot_funcs.keys():
            # 若是绘图模式则进行绘图可视化
            frame = self.plot_funcs[visual_mode](n_iter)
            # 若 frame 为 None 则不绘制
            if frame is None:
                return frame
            # 若是静态显示则进行静态绘制
            if static:
                frame.render_static(self.visual_config.get('fig_size', (6.4, 4.8)))
            else:
                # 检查迭代次数是否是空 或者 是否是最后一次迭代
                animate = False if n_iter is None or n_iter == self.max_iter else True
                self.animator.show(frame, animate)
        else:
            raise VisualizationError(f"Unrecognized visualization mode: {visual_mode}")
        return frame

    def plot_decisions(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        绘制决策空间实现可视化
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        xs = self.sols.xs if n_iter is None else self.history_sols[n_iter].xs
        frame = plot_decisions(xs, n_iter, **self.visual_config)
        return frame

    def plot_objectives(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        绘制目标空间实现可视化
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        if self.sols.n_objs == 1:  # 对于单目标问题绘制目标值矩阵历史记录情况
            history = self.history_objs if n_iter is None else self.history_objs[:n_iter + 1]
            frame = plot_history_objs(history, n_iter, **self.visual_config)
        else:  # 对于多目标问题则绘制目标值矩阵
            objs = self.sols.objs if n_iter is None else self.history_sols[n_iter].objs
            frame = plot_objectives(np.asarray(objs), n_iter, **self.visual_config)
        return frame

    def plot_hybrids_2d(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        实现决策空间与目标空间混合绘制(绘制二维图像)(仅支持单目标问题)
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        xs = self.sols.xs if n_iter is None else self.history_sols[n_iter].xs
        objs = self.sols.objs if n_iter is None else self.history_sols[n_iter].objs
        frame = plot_hybrids_2d(self.problem, xs, objs, n_iter, **self.visual_config)
        return frame

    def plot_hybrids_3d(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        实现决策空间与目标空间混合绘制(绘制三维图像)(仅支持单目标问题)
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        xs = self.sols.xs if n_iter is None else self.history_sols[n_iter].xs
        objs = self.sols.objs if n_iter is None else self.history_sols[n_iter].objs
        frame = plot_hybrids_3d(self.problem, xs, objs, n_iter, **self.visual_config)
        return frame

    def plot_metrics(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        根据历史记录绘制性能指标收敛变化
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        history_best = self.history_best if n_iter is None else self.history_best[:n_iter + 1]
        history_metrics = {key: [best.metrics[key] for best in history_best] for key in self.best.metrics}
        frame = plot_metrics(history_metrics, n_iter, **self.visual_config)
        return frame

    def plot_by_algorithm(self, n_iter: Optional[int] = None, **kwargs) -> Optional[Frame]:
        """
        提供算法自定义可视化的接口（算法子类可覆写）
        :param n_iter: 当前迭代次数，显示在标题中
        :param kwargs: 可指定参数（子类可继承与覆写）
        :return: 绘图帧实例 (失败则为None)
        """
        pass

    def plot_by_problem(self, n_iter: Optional[int] = None, **kwargs) -> Optional[Frame]:
        """
        提供问题自定义可视化的接口（问题子类可覆写）
        :param n_iter: 当前迭代次数，显示在标题中
        :param kwargs: 可指定参数（子类可继承与覆写）
        :return: 绘图帧实例 (失败则为None)
        """
        best = self.get_best(n_iter)
        if best is None:
            return None
        frame = self.problem.plot_by_problem(n_iter=n_iter, best=best, **kwargs)
        return frame

    def get_best(self, n_iter: Optional[int] = None, weight: Optional[Union[list, np.ndarray]] = None) -> 'Solutions':
        """
        获取最优解(集)，可选择指定迭代次数，对于多目标问题可按权重筛选最接近的解。
        :param n_iter: 第n次迭代的最优解(集)，若为None则取最后一次迭代结果
        :param weight: 多目标问题的偏好权重向量，例如 [0.3, 0.7]，长度需等于目标数
        :return: 符合条件的最优解(集)，单目标时直接返回最优，多目标时可选择返回整个解集或者最接近权重的解
        """
        if len(self.history_best):  # 先检查是否有历史最优解记录
            best = self.history_best[-1] if n_iter is None else self.history_best[n_iter]
        else:  # 若无历史最优解则默认使用当前最优解
            best = self.sols.get_best() if self.best is None else self.best
        # 多目标 + 指定权重 则 从候选解中挑选最接近权重的解
        if weight is not None and self.problem.n_objs > 1:
            # 截取有效维度，避免权重长度超出目标数
            weight = np.array(weight[:self.problem.n_objs], dtype=float)
            # 处理权重全零的边界情况（全零无法计算方向相似性）
            weight_norm = np.linalg.norm(weight)
            if weight_norm == 0:
                # 权重无效时退化为等权重
                weight = np.ones(self.problem.n_objs) / self.problem.n_objs
                weight_norm = 1.0
            # 归一化权重向量
            weight_unit = weight / weight_norm
            # 提取所有候选解的目标值矩阵 (n_bests x n_objs)
            best_objs = best.objs.copy()
            # 归一化每个解的目标向量（L2归一化，逐行处理）
            obj_norms = np.linalg.norm(best_objs, axis=1, keepdims=True)  # (n_bests, 1)
            obj_norms = np.maximum(obj_norms, 1e-8)  # 避免除零，保护数值稳定性
            objs_matrix_normed = best_objs / obj_norms  # (n_bests, n_objs)
            # 计算余弦相似度（批量点积）
            similarities = objs_matrix_normed @ weight_unit  # (n_bests,)
            # 找出最相似解的索引
            best_idx = np.argmax(similarities)
            # 提取对应的最优解
            best = best[best_idx]
        # 对最优解进行格式化以保证正确输出
        return self.finalize(best)

    def get_metric_value(self, metric_key: str):
        """
        获取算法在指定指标上的最终评估值

        该值可能来源于：
            - 最优解的目标值 (obj/con/front)
            - 解集的其他指标 (hv/gd/igd)
            - 算法运行时间 (time)
        :param metric_key: 指标键名
        :return: 指标数值，若无效则返回 NaN
        """
        best = self.get_best()
        # 根据给定指标关键字获取指标值
        if metric_key.lower() == 'obj':
            value = best.f
        elif metric_key.lower() == 'con':
            value = best.c
        elif metric_key.lower() == 'front':
            value = best.n_sols
        elif (metric_key.lower() == 'time' or
              metric_key.lower() == 'time(s)'):
            value = self.run_time
        else:
            value = best.get_metric(metric_key)
        # 确保指标值不为空
        value = float('nan') \
            if value is None else float(value)
        return value

    def save_sols(self,
                  file_path: Optional[str] = None,
                  file_format: Optional[str] = None,
                  as_object: bool = False):
        """
        保存当前解
        :param file_path: 文件路径，默认当前路径，保存文件名为{算法名}_sols_{时间戳}
        :param file_format: 文件格式 'csv', 'json', 'pkl', 'npz', 默认csv
        :param as_object: 仅对pkl有效，True保存完整对象，False保存dict
        """
        # 生成默认路径
        if file_path is None:
            timestamp = datetime.now().strftime("%m%d%H%M%S%f")[:-3]
            file_path = os.path.abspath(f"{type(self).__name__}_sols_{timestamp}")
        self.sols.save(file_path, file_format, as_object)

    def save_best(self,
                  file_path: Optional[str] = None,
                  file_format: Optional[str] = None,
                  as_object: bool = False,
                  weight: Optional[Union[list, np.ndarray]] = None):
        """
        保存当前最优解
        :param file_path: 文件路径，默认当前路径，保存文件名为{算法名}_best_{时间戳}
        :param file_format: 文件格式 'csv', 'json', 'pkl', 'npz', 默认csv
        :param as_object: 仅对pkl有效，True保存完整对象，False保存dict
        :param weight: 多目标问题的偏好权重向量，例如 [0.3, 0.7]，长度需等于目标数
        """
        best = self.get_best(weight=weight) if self.best is None or weight is not None else self.best
        # 生成默认路径
        if file_path is None:
            timestamp = datetime.now().strftime("%m%d%H%M%S%f")[:-3]
            file_path = os.path.abspath(f"{type(self).__name__}_best_{timestamp}")
        best.save(file_path, file_format, as_object)

    def save_history(self,
                     folder_path: Optional[str] = None,
                     file_format: Optional[str] = None,
                     as_object: bool = False,
                     best_only: bool = False):
        """
        保存所有历史解
        :param folder_path: 文件夹路径，默认当前路径，保存文件名为{算法名}_history_sols_{时间戳}
        :param file_format: 文件格式 'csv', 'json', 'pkl', 'npz', 默认csv
        :param as_object: 仅对pkl有效，True保存完整对象，False保存dict
        :param best_only: 是否只保存历史最优解，默认为否
        """
        if folder_path is None:
            timestamp = datetime.now().strftime("%m%d%H%M%S%f")[:-3]
            folder_path = os.path.abspath(f"{type(self).__name__}_history_sols_{timestamp}")
        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)
        # 确定文件扩展名（csv是文件夹，无扩展名）
        ext_map = {'csv': '', 'npz': '.npz', 'json': '.json', 'pkl': '.pkl'}
        # 检查是否只保存历史最优解
        history = self.history_best if best_only else self.history_sols
        # 遍历保存所有历史解
        for i, sol in enumerate(history):
            if sol is None:
                continue
            # 构建文件路径
            if file_format == 'csv':
                # csv格式：创建子文件夹
                sub_path = os.path.join(
                    folder_path, f"iter_{i:0{len(str(self.max_iter))}d}"
                )
            else:
                # 其他格式：直接保存文件
                sub_path = os.path.join(
                    folder_path, f"iter_{i:0{len(str(self.max_iter))}d}{ext_map.get(str(file_format), '')}"
                )
            # 保存
            sol.save(sub_path, file_format=file_format, as_object=as_object)

    def get_result_info(self, float_format: str = ".6e") -> str:
        """
        获取算法优化结果信息(返回字符串)
        :param float_format: 浮点数的格式化格式，默认为 ".6e"
        :return: 优化结果字符串
        """
        # 获取最优解
        best = self.get_best()
        # 设置分隔符数量
        num_sep = 66
        lines = list()
        lines.append("=" * num_sep)
        lines.append("OPTIMIZATION RESULT" + f" - {type(self).__name__}")
        lines.append("=" * num_sep)
        # 基础信息
        lines.append(f"Iterations: {self.n_iter}")
        lines.append(f"Runtime: {self.run_time:.6f} s")
        lines.append(f"Number of Bests: {best.n_sols}")
        # 指标信息
        for key in best.metrics:
            lines.append(f"Best {key}: {best.get_metric(key):{float_format}}")
        # 目标/约束/决策变量信息
        lines.append(best.format_matrix_info("Best Objectives", best.f, best.n_objs, float_format))
        lines.append(best.format_matrix_info("Best Constraints", best.c, best.n_cons, float_format))
        lines.append(best.format_matrix_info("Best Decision Variables", best.x, best.n_vars, float_format))
        lines.append("=" * num_sep)
        return "\n".join(lines)

    def report_result(self, float_format: str = ".6e"):
        """
        报告算法优化结果信息
        :param float_format: 浮点数的格式化格式，默认为 ".6e"
        """
        print(self.get_result_info(float_format))

    def set_log_file(self, file_path: Optional[str] = None) -> None:
        """
        设置日志文件路径
        :param file_path: 默认日志路径为当前路径，命名为{算法名}_{时间戳}.log
        """
        if VisualMode.parse(self.visual_mode) != VisualMode.LOG:
            raise ValueError("File logging requires LOG visual mode")
        # 生成默认路径
        if file_path is None:
            timestamp = datetime.now().strftime("%m%d%H%M%S%f")[:-3]
            file_path = f"{type(self).__name__}_{timestamp}.log"
            self.logger = setup_logger(os.path.abspath(file_path), to_file=True)
        else:
            self.logger = setup_logger(file_path, to_file=True)

    def format_log_line(self) -> str:
        """格式化当前迭代日志行"""
        # 获取当前最优解
        best = self.get_best()
        if not best:
            return f"[{type(self).__name__}] No solution"
        # 可行解比例
        if self.sols.cons is not None and self.sols.n_cons > 0:
            feasible = np.all(self.sols.cons <= 0, axis=1)
            feasible_ratio = np.mean(feasible) * 100
        else:
            feasible_ratio = 100.0

        lines = list()
        prefix = f"[{type(self).__name__}] "
        # 迭代次数信息
        lines.append(f"Iter: {self.n_iter:0{len(str(self.max_iter))}d}/{self.max_iter}")

        # 单目标优化相关信息
        if self.problem.n_objs == 1:
            obj_value = float(best.f) if best.f is not None else float('nan')
            lines.append(f"Obj: {obj_value:.6e}")
            # 如果有约束再加入约束信息
            if best.cons is not None and best.n_cons > 0:
                con_value = float(best.c) if best.c is not None else float('nan')
                lines.insert(2, f"Con: {con_value:.6e}")
        # 多目标优化相关信息
        else:
            lines.append(f"Front: {len(best):{len(str(self.n_sols))}d}")

        # 加入指标信息
        for key in best.metrics:
            lines.append(f"{key}: {best.get_metric(key):.6e}")
        # 可行解比例信息
        lines.append(f"Feas: {feasible_ratio:.1f} %")
        # 运行时间信息
        lines.append(f"Time: {self.run_time:.3f} s")

        return prefix + " | ".join(lines)

    def get_config(self) -> dict:
        """获取算法的完整配置（用于保存和复现状态）"""
        return {
            # 版本数据
            'version': __version__,
            # 算法名称
            'algorithm': self.__class__.__name__,
            # 核心控制参数
            'num_solutions': self.n_sols,
            'max_iterations': self.max_iter,
            # 遗传算子参数
            'crossover_probability': self.cross_prob,
            'mutation_probability': self.mutate_prob,
            # 支持的目标个数（是否只支持单目标）
            'single_obj_only': self.single_obj_only,
            # 支持的问题变量类型（返回整数列表）
            'supported_var_types': [var_type.value for var_type in self.supported_var_types]
        }
