# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
算法评估器
"""

import os
import copy
import numpy as np
from tqdm import tqdm
from ..problems import Problem
from .algorithm import Algorithm
from scipy.stats import mannwhitneyu
from ..types import VisualMode, MetricType
from ..utilities.visualization import Frame
from typing import Union, Optional, List, Dict, Tuple
from ..core import warn_once, AlgorithmError, ProblemError
from ..utilities.handler import format_as_table, save_as_table


class Evaluator(object):
    def __init__(self,
                 problems: Union[List[Problem], Dict[str, Problem]],
                 algorithms: Union[List[Algorithm], Dict[str, Algorithm]],
                 n_runs: int = 5,
                 n_sols: Optional[int] = None,
                 same_start: bool = False):
        """
        算法评估器(多问题多算法评估对比)
        :param problems: 问题实例
        :param algorithms: 对比的算法集合(字典或列表)
        :param n_runs: 每种算法的运行次数(必须指定)
        :param n_sols: 解集大小(初始化相同的解时有效)
        :param same_start: 是否初始化相同的解
        """
        # 初始化问题实例集合（列表自动转换为字典）
        self.problems = self._list_to_dict(problems)
        # 初始化算法实例集合（列表自动转换为字典）
        self.algorithms = self._list_to_dict(algorithms)
        self.n_runs = n_runs
        self.n_sols = n_sols
        self.same_start = same_start
        # 记录迭代次数
        self.n_iter = 0
        # 保存追踪指标集合
        self.metrics_set = []
        # 问题标签：是否均是多目标问题
        self._is_multi_objs = None
        # 初始化评估任务(问题-算法对)
        self.eval_tasks = dict()
        # 若没有指定解集大小则默认使用第一个算法的解集大小参数
        self.n_sols = next(iter(self.algorithms.values())).n_sols \
            if self.n_sols is None else self.n_sols

    @staticmethod
    def _list_to_dict(value):
        """将列表转换为字典格式"""
        if isinstance(value, dict):
            return value
        elif isinstance(value, list):
            return {type(item).__name__: item for item in value}
        else:
            raise TypeError(f"Expected dict or list, got {type(value).__name__}")

    def _verify_consistency(self):
        """验证所有实例的一致性"""
        # 验证问题的一致性
        # 检查是否均是单目标问题
        single_obj_problem = [problem.n_objs == 1 for problem in self.problems.values()]
        # 检查是否均是一类问题（全是单目标或全是多目标），若不是则报错
        if len(set(single_obj_problem)) > 1:
            raise ProblemError(
                f"Inconsistent problem types: {list(self.problems.keys())} contains "
                f"both single-objective and multi-objective problems. "
                f"All problems must be either single-objective or multi-objective."
            )
        # 判断整体问题是单目标还是多目标
        self._is_multi_objs = not all(single_obj_problem)
        # 检查所有算法是否只能求解单目标问题
        single_obj_only_algorithms = [
            algorithm_name for algorithm_name, algorithm in self.algorithms.items()
            if getattr(algorithm, 'single_obj_only', False)
        ]
        # 多目标算法默认可以求解单目标问题，但单目标算法不可以求解多目标问题
        # 若问题均是多目标问题，但算法存在只能求解单目标的算法则报错
        if self._is_multi_objs and single_obj_only_algorithms:
            raise AlgorithmError(
                f"Multi-objective problem(s) {list(self.problems.keys())} incompatible with "
                f"single-objective-only algorithm(s): {single_obj_only_algorithms}. "
                f"Remove these algorithms or use multi-objective compatible ones."
            )

    def _init_evaluator(self):
        """初始化比较器"""
        # 先验证所有实例的一致性
        self._verify_consistency()
        # 初始化所有评估任务(问题-算法对)
        self.eval_tasks = dict()
        # 遍历所有需要优化的问题
        for (problem_name, problem) in self.problems.items():
            # 初始化问题对应算法字典
            self.eval_tasks[problem_name] = dict()
            # 若使用相同解集初始化则先初始化解集
            random_sols = problem.init_decs_mat(self.n_sols) \
                if self.same_start else None and self.n_sols is not None
            # 初始化所有算法
            for (algorithm_name, algorithm_template) in self.algorithms.items():
                # 深拷贝，确保不修改算法原来的参数
                algorithm_base = copy.deepcopy(algorithm_template)
                # 拷贝原始算法后，初始化算法
                # noinspection PyProtectedMember
                algorithm_base._init_algorithm(problem)
                # 若使用相同初始化则使用一样的解
                # noinspection PyProtectedMember
                algorithm_base._init_solutions(seeds=random_sols)
                # 将算法内部设置的可视化模式置空
                algorithm_base.visual_mode = VisualMode.NONE
                # 更新追踪指标的交集数据
                # noinspection PyProtectedMember
                for metric in algorithm_base._metrics_set:
                    if metric not in self.metrics_set:
                        self.metrics_set.append(metric)
                # 创建多次运行的独立实例以实现运行 n_runs 次
                self.eval_tasks[problem_name][algorithm_name] = [
                    copy.deepcopy(algorithm_base) for _ in range(self.n_runs)
                ]

    def track_metrics(self, metrics: Optional[Union[str, list, MetricType]] = None):
        """
        :param metrics: 设置要追踪的性能指标
        """
        for algorithm in self.algorithms.values():
            algorithm.track_metrics(metrics)

    def run_evaluation(self):
        """运行评估器"""
        self._init_evaluator()
        # 逐个问题求解
        for (problem_name, algorithm_map) in self.eval_tasks.items():
            # 逐个算法运行
            for (algorithm_name, algorithm_instances) in algorithm_map.items():
                print(f"{algorithm_name} optimize {problem_name}...")
                # 默认直接展示进度
                for i in tqdm(range(self.n_runs)):
                    algorithm_instances[i].run()

    def plot_violin(self,
                    metric_name=None,
                    problem_name=None,
                    fig_size=(6.4, 4.8),
                    inner="box",
                    **kwargs):
        """
        绘制指定问题的小提琴图以对比算法效果
        :param metric_name: 指标名称（如 'obj', 'gd', 'hv'）
                            - 若为 None，单目标问题默认使用 'obj'
                            - 多目标问题且无追踪指标时默认使用 'front'
                            - 多目标问题且有追踪指标时默认使用第一个指标
        :param problem_name: 问题名称
                            - 若为 None，默认使用第一个问题
        :param fig_size: 绘图的大小
        :param inner: 控制小提琴内部显示的内容
        :return: 绘图帧实例
        """
        frame = Frame()
        # 确定指标与问题名称
        metric_name = self._resolve_metric_name(metric_name)
        problem_name = self._resolve_problem_name(problem_name)
        # 提取数据
        data = self.get_problem_metric(metric_name, problem_name)
        # 添加图像
        frame.add_sns_violin(data=data, inner=inner, **kwargs)
        frame.set_labels(xlabel='Algorithms', ylabel=metric_name)
        frame.set_title(problem_name)
        # 绘制图像
        frame.render_static(fig_size)
        return frame

    def plot_box(self,
                 metric_name=None,
                 problem_name=None,
                 fig_size=(6.4, 4.8),
                 **kwargs):
        """
        绘制指定问题的箱线图以对比算法效果
        :param metric_name: 指标名称（如 'obj', 'gd', 'hv'）
                            - 若为 None，单目标问题默认使用 'obj'
                            - 多目标问题且无追踪指标时默认使用 'front'
                            - 多目标问题且有追踪指标时默认使用第一个指标
        :param problem_name: 问题名称
                            - 若为 None，默认使用第一个问题
        :param fig_size: 绘图的大小
        :return: 绘图帧实例
        """
        frame = Frame()
        # 确定指标与问题名称
        metric_name = self._resolve_metric_name(metric_name)
        problem_name = self._resolve_problem_name(problem_name)
        # 提取数据
        data = self.get_problem_metric(metric_name, problem_name)
        # 添加图像
        frame.add_sns_box(data=data, **kwargs)
        frame.set_labels(xlabel='Algorithms', ylabel=metric_name, **kwargs)
        frame.set_title(problem_name)
        # 绘制图像
        frame.render_static(fig_size)
        return frame

    def plot_kde(self,
                 metric_name=None,
                 problem_name=None,
                 fig_size=(6.4, 4.8),
                 **kwargs):
        """
        绘制指定问题的核密度估计图以对比算法效果
        :param metric_name: 指标名称（如 'obj', 'gd', 'hv'）
                            - 若为 None，单目标问题默认使用 'obj'
                            - 多目标问题且无追踪指标时默认使用 'front'
                            - 多目标问题且有追踪指标时默认使用第一个指标
        :param problem_name: 问题名称
                            - 若为 None，默认使用第一个问题
        :param fig_size: 绘图的大小
        :return: 绘图帧实例
        """
        frame = Frame()
        # 确定指标与问题名称
        metric_name = self._resolve_metric_name(metric_name)
        problem_name = self._resolve_problem_name(problem_name)
        # 提取数据
        data = self.get_problem_metric(metric_name, problem_name)
        # 添加图像
        frame.add_sns_kde(data=data, fill=True)
        frame.set_labels(xlabel=metric_name, ylabel='Density', **kwargs)
        frame.set_title(problem_name)
        # 绘制图像
        frame.render_static(fig_size)
        return frame

    def get_problem_metric(self, metric_name: str, problem_name: str):
        """
        提取指定问题和指标的评估数据

        该函数用于从完整的评估结果中，提取某个指标在某个问题上的所有算法数据。
        如果未指定指标或问题，会使用合理的默认值

        :param metric_name: 指标名称（必须指定）
        :param problem_name: 问题名称（必须指定）
        :return: 算法数据字典，格式为：
                    {
                        'Algorithm_A': [value1, value2, ...],  # 多次运行的结果列表
                        'Algorithm_B': [value3, value4, ...],
                        ...
                    }
        """
        # 获取指标数据
        metric_data = self.get_metric_data(metric_name)

        # 验证是否有数据存在
        if not metric_data or 'Problem' not in metric_data:
            raise ValueError("No problem data available in metric_data")

        # 验证问题是否存在
        if problem_name not in metric_data['Problem']:
            raise KeyError(
                f"Problem '{problem_name}' not found. "
                f"Available problems: {metric_data['Problem']}"
            )

        # 提取该问题的所有算法数据
        result = {}
        problem_index = metric_data['Problem'].index(problem_name)

        for key, values in metric_data.items():
            if key != 'Problem':
                result[key] = values[problem_index]

        return result

    def _resolve_metric_name(self, metric_name: Optional[str] = None) -> str:
        """
        解析并返回有效的指标名称

        如果未指定指标名称，根据问题类型自动选择默认指标：
            - 单目标问题：返回 'obj'
            - 多目标问题且有追踪指标：返回第一个追踪指标
            - 多目标问题且无追踪指标：返回 'front' 并发出警告
        :param metric_name: 指定的指标名称，可为 None
        :return: 有效的指标名称字符串
        """
        if metric_name is not None:
            return metric_name

        if not self._is_multi_objs:
            return 'obj'

        if len(self.metrics_set) == 0:
            warn_once(
                "No tracking metrics configured. Falling back to Pareto front size, "
                "which may not provide meaningful algorithm comparison results. "
                "Please use track_metrics() to specify metrics (e.g., track_metrics('hv')) "
                "for proper evaluation."
            )
            return 'front'

        return self.metrics_set[0]

    def _resolve_problem_name(self, problem_name: Optional[str] = None) -> str:
        """
        解析并返回有效的问题名称；
        如果未指定问题名称，返回第一个可用的问题名称。
        :param problem_name: 指定的问题名称，可为 None
        :return: 有效的问题名称字符串
        """
        # 如果已指定，直接返回（由调用方负责验证）
        if problem_name is not None:
            return problem_name

        # 获取默认问题（第一个）
        if not self.problems:
            raise ValueError("No problems available. Cannot resolve problem name.")

        return next(iter(self.problems.keys()))

    def get_result_data(self, mean_format: str = ".6e", var_format: str = ".2e", enable_stats_test: bool = False):
        """
        获取整个评估任务的结果数据
        :param mean_format: 均值的格式化格式，默认为 ".6e"
        :param var_format: 方差的格式化格式，默认为 ".2e"
        :param enable_stats_test: 是否启用统计检验
        :return: 结果数据
        """
        result_data = dict()
        # 如果问题均是单目标问题
        if not self._is_multi_objs:
            result_data['Obj'] = self.get_metric_data(
                'Obj',
                data_format='string',
                mean_format=mean_format,
                var_format=var_format,
                enable_stats_test=enable_stats_test,
            )
            result_data['Con'] = self.get_metric_data(
                'Con',
                data_format='string',
                mean_format=mean_format,
                var_format=var_format,
                enable_stats_test=enable_stats_test,
            )
        # 如果问题均是多目标问题
        else:
            # 如果没有指标则默认使用前沿解数量，并报出警告
            if len(self.metrics_set) == 0:
                warn_once(
                    "No tracking metrics configured. Falling back to Pareto front size, "
                    "which may not provide meaningful algorithm comparison results. "
                    "Please use track_metrics() to specify metrics (e.g., track_metrics('hv')) "
                    "for proper evaluation."
                )
                result_data['Front'] = self.get_metric_data(
                    'front',
                    data_format='string',
                    mean_format=mean_format,
                    var_format=var_format,
                    enable_stats_test=enable_stats_test,
                )
        # 若存在追踪指标则获取追踪指标数据
        for metric in self.metrics_set:
            result_data[metric] = self.get_metric_data(
                metric,
                data_format='string',
                mean_format=mean_format,
                var_format=var_format,
                enable_stats_test=enable_stats_test,
            )
        # 加入追踪时间指标数据
        result_data['Time(s)'] = self.get_metric_data(
            'Time(s)',
            data_format='string',
            mean_format=mean_format,
            var_format=var_format,
            enable_stats_test=enable_stats_test,
        )
        return result_data

    def report_result(self,
                      mean_format: str = ".6e",
                      var_format: str = ".2e",
                      transpose: bool = False,
                      enable_stats_test: bool = True,
                      save_dir: Optional[str] = None):
        """
        报告算法优化结果信息，并可选保存到指定文件夹(保存为csv文件)
        :param mean_format: 均值的格式化格式，默认为 ".6e"
        :param var_format: 方差的格式化格式，默认为 ".2e"
        :param transpose: 是否将行与列转置（即将整个表格进行转置）
        :param enable_stats_test: 是否启用统计检验
        :param save_dir: 文件夹保存路径，如果为None则只打印表格，不保存文件
        """
        result_data = self.get_result_data(
            mean_format, var_format, enable_stats_test
        )
        for metric_key, metric_data in result_data.items():
            table_info = format_as_table(
                data=metric_data,
                row_key='Problem',
                col_key='Algorithm',
                transpose=transpose
            )
            print(f"\n*** Metric: {metric_key} ***")
            print(table_info)
            if save_dir:
                # 确保目录存在
                os.makedirs(save_dir, exist_ok=True)
                csv_file = os.path.join(save_dir, f"{metric_key}.csv")
                save_as_table(
                    data=result_data,
                    csv_path=csv_file,
                    row_key='Problem',
                    col_key='Algorithm',
                    transpose=transpose
                )

    def get_metric_data(self,
                        metric_key: str,
                        data_format: str = 'raw',
                        mean_format: str = ".6e",
                        var_format: str = ".2e",
                        enable_stats_test: bool = False) -> Dict[str, List]:
        """
        获取指定追踪指标的评估结果，返回按列组织的表格数据

        :param metric_key: 指标键名（如 'obj', 'hv', 'gd'）
        :param data_format: 数据返回格式
                - 'raw': 返回原始多次运行数据列表
                - 'stats': 返回统计值 (mean, variance)
                - 'string': 返回格式化字符串 "均值(方差)"
        :param mean_format: 均值的格式化格式，默认为 ".6e"
        :param var_format: 方差的格式化格式，默认为 ".2e"
        :param enable_stats_test: 是否启用统计检验（仅在 data_format='string' 时生效）
        :return: 按列组织的表格数据，格式为：
            {
                'Problem': ['P1', 'P2', ...],
                'Algorithm_A': [value1, value2, ...],
                'Algorithm_B': [value3, value4, ...],
                ...
            }
        """
        # 参数验证
        valid_formats = ['raw', 'stats', 'string']
        if data_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid data_format: '{data_format}'. "
                f"Must be one of: {valid_formats}"
            )
        if enable_stats_test and data_format.lower() != 'string':
            raise ValueError(
                f"Statistical test can only be enabled when data_format='string'. "
                f"Current data_format='{data_format}'"
            )

        # 初始化结果字典，首列为问题名称
        result = {'Problem': []}

        # 获取所有算法名称（保持顺序）
        all_algorithms = []
        for algorithm_map in self.eval_tasks.values():
            for algo_name in algorithm_map.keys():
                if algo_name not in all_algorithms:
                    all_algorithms.append(algo_name)

        # 为每个算法初始化空列表
        for algorithm_name in all_algorithms:
            result[algorithm_name] = []

        # 逐个问题处理数据
        for problem_name, algorithm_map in self.eval_tasks.items():
            # 添加问题名称到首列
            result['Problem'].append(problem_name)
            # 如果启用统计检验，预先计算基准算法（最后一个算法）的数据
            baseline_name = None
            baseline_values = None
            if enable_stats_test and algorithm_map:
                baseline_name = list(algorithm_map.keys())[-1]
                baseline_instances = list(algorithm_map.values())[-1]
                baseline_values = np.array([algorithm.get_metric_value(metric_key)
                                            for algorithm in baseline_instances])
            # 逐个算法获取数据
            for algorithm_name, algo_instances in algorithm_map.items():
                # 收集多次运行的结果
                run_values = []
                for algo_instance in algo_instances:
                    try:
                        value = algo_instance.get_metric_value(metric_key)
                        # 处理 None 值
                        if value is None:
                            value = float('nan')
                        run_values.append(float(value))
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to get metric '{metric_key}' for "
                            f"problem '{problem_name}', algorithm '{algorithm_name}': {e}"
                        )
                # 根据 data_format 处理数据
                processed_value = self._process_metric_data(
                    run_values=run_values,
                    data_format=data_format,
                    mean_format=mean_format,
                    var_format=var_format,
                    enable_stats_test=enable_stats_test,
                    baseline_values=baseline_values if algorithm_name != baseline_name else None,
                    metric_key=metric_key
                )
                # 添加到结果列
                result[algorithm_name].append(processed_value)

        return result

    def _process_metric_data(self,
                             metric_key: str,
                             run_values: List[float],
                             data_format: str,
                             mean_format: str,
                             var_format: str,
                             enable_stats_test: bool,
                             baseline_values: Optional[np.ndarray]
                             ) -> Union[List[float], Tuple[float, float], str]:
        """
        处理单组运行数据的内部方法

        :param metric_key: 指标键名（如 'obj', 'hv', 'gd'）
        :param run_values: 单组运行数据
        :param data_format: 数据返回格式
                - 'raw': 返回原始多次运行数据列表
                - 'stats': 返回统计值 (mean, variance)
                - 'string': 返回格式化字符串 "均值(方差)"
        :param mean_format: 均值的格式化格式，默认为 ".6e"
        :param var_format: 方差的格式化格式，默认为 ".2e"
        :param enable_stats_test: 是否启用统计检验（仅在 data_format='string' 时生效）
        :param baseline_values: 基准算法的数据
        :return: 根据 data_format 返回不同格式的数据
        """
        # 转换为 numpy 数组以便计算
        values = np.array(run_values)

        # 处理全为 NaN 的情况
        if np.all(np.isnan(values)):
            if data_format == 'raw':
                return run_values
            elif data_format == 'stats':
                return float('nan'), float('nan')
            else:  # string
                return "NaN"

        # 计算统计量
        mean = np.nanmean(values)  # 使用 nanmean 忽略 NaN
        var = np.nanvar(values, ddof=1) if len(values) > 1 else 0.0  # 样本方差

        # 根据格式返回
        if data_format == 'raw':
            return run_values

        elif data_format == 'stats':
            return mean, var

        else:  # data_format == 'string'
            # 格式化均值和方差
            mean_str = f"{mean:{mean_format}}"
            var_str = f"{var:{var_format}}"

            # 处理特殊值
            if np.isnan(mean) or np.isnan(var):
                result_str = "NaN"
            else:
                result_str = f"{mean_str}({var_str})"

            # 添加统计检验结果
            if enable_stats_test and baseline_values is not None:
                # 判断指标是否越小越好（HV 越大越好，其他越小越好）
                lower_is_better = False if metric_key.lower() == 'hv' else True
                comparison = self._compare_with_baseline(
                    values, baseline_values, lower_is_better
                )
                result_str += comparison

            return result_str

    @staticmethod
    def _compare_with_baseline(current_values: np.ndarray,
                               baseline_values: np.ndarray,
                               lower_is_better: bool = True,
                               alpha: float = 0.05) -> str:
        """
        使用 Mann-Whitney U 检验比较当前算法与基准算法的优劣

        :param current_values: 当前算法的运行结果数组
        :param baseline_values: 基准算法的运行结果列表
        :param lower_is_better: 指标是否越小越好
        :param alpha: 
        :return: '+' : 当前算法显著优于基准算法，
                 '-' : 当前算法显著劣于基准算法，
                 '=' : 无显著差异
        """
        # 移除 NaN 值
        current_clean = current_values[~np.isnan(current_values)]
        baseline_clean = baseline_values[~np.isnan(baseline_values)]

        # 如果数据不足，返回 '='
        if len(current_clean) < 2 or len(baseline_clean) < 2:
            return '='

        try:
            # 执行 Mann-Whitney U 检验
            u_stat, p_value = mannwhitneyu(
                current_clean, baseline_clean,
                alternative='two-sided',
                method='auto'
            )

            # 如果 p 值大于显著性水平，无显著差异
            if p_value >= alpha:
                return '='

            # 根据 U 统计量判断方向
            total_pairs = len(current_clean) * len(baseline_clean)
            if lower_is_better:
                # 越小越好：U 值小说明当前算法更优
                return '+' if u_stat < total_pairs / 2 else '-'
            else:
                # 越大越好：U 值大说明当前算法更优
                return '+' if u_stat > total_pairs / 2 else '-'

        except Exception as e:
            # 统计检验失败时返回 '='
            warn_once(f"Mann-Whitney U test failed: {str(e)}. Returning '=' as fallback.")
            return '='
