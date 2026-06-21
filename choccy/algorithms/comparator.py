# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
算法比较器
"""

from tqdm import tqdm
from .algorithm import Algorithm
from ..types import VisualMode, MetricType
from ..utilities.logging import setup_logger
from ..utilities.visualization import Animator
from ..utilities.visualization.comparison import *
from ..core import warn_once, VisualizationError
from typing import Union, Optional, List, Dict
from ..utilities.handler import format_as_table, save_as_table


class Comparator(object):
    def __init__(self,
                 problem: Problem,
                 algorithms: Union[List[Algorithm], Dict[str, Algorithm]],
                 n_sols: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 same_start: bool = False,
                 visual_mode: Optional[str] = None):
        """
        算法比较器(用于实时对比多个算法效果)
        :param problem: 问题实例
        :param algorithms: 对比的算法集合(字典或列表)
        :param n_sols: 解集大小 (初始化相同的解集时有效)
        :param max_iter: 迭代次数 (若为空则按照全局最大)
        :param same_start: 是否初始化相同的解集
        :param visual_mode: 可视化模式
        """
        # 初始化问题实例
        self.problem = problem
        # 初始化算法实例集合（列表自动转换为字典）
        self.algorithms = self._list_to_dict(algorithms)
        self.n_sols = n_sols
        self.max_iter = max_iter
        self.same_start = same_start
        # 初始化可视化模式(默认进度条显示)
        self.visual_mode = 'progress' \
            if visual_mode is None else visual_mode
        # 记录迭代次数
        self.n_iter = 0
        # 保存追踪指标集合
        self.metrics_set = []
        # 若没有指定解集大小则默认使用第一个算法的解集大小参数
        self.n_sols = next(iter(self.algorithms.values())).n_sols \
            if self.n_sols is None else self.n_sols
        # 设置日志记录器
        self.logger = setup_logger()
        # 对可视化模式进行解析
        self.visual_mode = VisualMode.parse(self.visual_mode)
        # 初始化默认可视化配置
        self.visual_config = {
            'fig_size': (6.4, 4.8),
            'interval': 30,
            'pareto_front': self.problem.pareto_front,
        }
        # 初始化动画绘制器
        self.animator = Animator()
        # 初始化绘图函数映射
        self.plot_funcs = {
            VisualMode.METRICS: self.plot_metrics_comparison,
            VisualMode.DECISIONS: self.plot_decision_comparison,
            VisualMode.OBJECTIVES: self.plot_objectives_comparison,
            VisualMode.HYBRIDS_2D: self.plot_hybrids_2d_comparison,
            VisualMode.HYBRIDS_3D: self.plot_hybrids_3d_comparison,
        }

    @staticmethod
    def _list_to_dict(value):
        """将列表转换为字典格式"""
        if isinstance(value, dict):
            return value
        elif isinstance(value, list):
            return {type(item).__name__: item for item in value}
        else:
            raise TypeError(f"Expected dict or list, got {type(value).__name__}")

    def _init_comparator(self):
        """初始化比较器"""
        global_max_iter = 0  # 用于统计全局最大迭代次数
        # 若使用相同解集初始化则先初始化解集
        random_sols = self.problem.init_decs_mat(self.n_sols) \
            if self.same_start else None
        # 初始化所有算法
        for algorithm in self.algorithms.values():
            # 初始化算法
            # noinspection PyProtectedMember
            algorithm._init_algorithm(self.problem)
            # 若使用相同初始化则使用一样的解
            # noinspection PyProtectedMember
            algorithm._init_solutions(seeds=random_sols)
            # 将算法内部设置的可视化模式置空
            algorithm.visual_mode = VisualMode.NONE
            # 更新全局最大迭代次数
            global_max_iter = max(global_max_iter, algorithm.max_iter)
            # 更新追踪指标的交集数据
            # noinspection PyProtectedMember
            for metric in algorithm._metrics_set:
                if metric not in self.metrics_set:
                    self.metrics_set.append(metric)
        # 设置最大迭代次数
        self.max_iter = global_max_iter if self.max_iter is None else self.max_iter

    def track_metrics(self, metrics: Optional[Union[str, list, MetricType]] = None):
        """
        :param metrics: 设置要追踪的性能指标
        """
        for algorithm in self.algorithms.values():
            algorithm.track_metrics(metrics)

    def iterator(self):
        """构建迭代器"""
        if self.visual_mode == VisualMode.PROGRESS:
            return tqdm(range(1, self.max_iter + 1))
        else:
            return range(1, self.max_iter + 1)

    def run_comparison(self):
        """运行比较器"""
        self._init_comparator()
        # 完成所有算法的准备并记录初始状态
        for algorithm in self.algorithms.values():
            # noinspection PyProtectedMember
            algorithm._prepare()
            algorithm.record_state()
        # 绘制初始状态图
        self.plot(n_iter=self.n_iter, static=False)
        # 算法迭代并优化问题
        for self.n_iter in self.iterator():
            for algorithm in self.algorithms.values():
                if self.n_iter < algorithm.max_iter and not algorithm._should_stop():
                    # noinspection PyProtectedMember
                    algorithm._run_step(self.n_iter)
                # 记录解集状态
                algorithm.record_state()
            # 绘制迭代过程中每步状态
            self.plot(n_iter=self.n_iter, static=False)

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

    def plot_decision_comparison(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        绘制决策空间实现多个算法对比可视化
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        decs_dict = dict()
        for name, algorithm in self.algorithms.items():
            if n_iter is None:
                decs_dict[name] = algorithm.sols.xs
            else:
                decs_dict[name] = algorithm.history_sols[n_iter].xs
        frame = plot_decision_comparison(decs_dict, n_iter, **self.visual_config)
        return frame

    def plot_objectives_comparison(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        绘制目标空间实现多个算法对比可视化
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        # 对于单目标问题绘制目标值矩阵历史记录情况
        if self.problem.n_objs == 1:
            history_dict = dict()
            for name, algorithm in self.algorithms.items():
                if n_iter is None:
                    history_dict[name] = algorithm.history_objs
                else:
                    history_dict[name] = algorithm.history_objs[:n_iter + 1]
            frame = plot_history_comparison(history_dict, n_iter, **self.visual_config)
        else:  # 对于多目标问题则绘制目标值矩阵
            objs_dict = dict()
            for name, algorithm in self.algorithms.items():
                if n_iter is None:
                    objs_dict[name] = algorithm.sols.objs
                else:
                    objs_dict[name] = algorithm.history_sols[n_iter].objs
            frame = plot_objectives_comparison(objs_dict, n_iter, **self.visual_config)
        return frame

    def plot_hybrids_2d_comparison(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        实现决策空间与目标空间混合绘制，实现多个算法对比可视化(绘制二维图像)(仅支持单目标问题)
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        decs_dict = dict()
        objs_dict = dict()
        for name, algorithm in self.algorithms.items():
            if n_iter is None:
                decs_dict[name] = algorithm.sols.xs
                objs_dict[name] = algorithm.sols.objs
            else:
                decs_dict[name] = algorithm.history_sols[n_iter].xs
                objs_dict[name] = algorithm.history_sols[n_iter].objs
        frame = plot_hybrids_2d_comparison(self.problem, decs_dict, objs_dict, n_iter, **self.visual_config)
        return frame

    def plot_hybrids_3d_comparison(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        实现决策空间与目标空间混合绘制，实现多个算法对比可视化(绘制二维图像)(仅支持单目标问题)
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        decs_dict = dict()
        objs_dict = dict()
        for name, algorithm in self.algorithms.items():
            if n_iter is None:
                decs_dict[name] = algorithm.sols.xs
                objs_dict[name] = algorithm.sols.objs
            else:
                decs_dict[name] = algorithm.history_sols[n_iter].xs
                objs_dict[name] = algorithm.history_sols[n_iter].objs
        frame = plot_hybrids_3d_comparison(self.problem, decs_dict, objs_dict, n_iter, **self.visual_config)
        return frame

    def plot_metrics_comparison(self, n_iter: Optional[int] = None) -> Optional[Frame]:
        """
        根据历史记录绘制性能指标收敛变化，实现多个算法对比可视化（默认绘制第一个追踪指标）
        :param n_iter: 当前迭代次数，显示在标题中
        :return: 绘图帧实例 (失败则为None)
        """
        # 默认绘制第一个追踪指标
        key = self.metrics_set[0]
        history_metrics = dict()
        for name, algorithm in self.algorithms.items():
            if n_iter is None:
                history_best = algorithm.history_best
            else:
                history_best = algorithm.history_best[:n_iter + 1]
            history_metrics[name] = [best.metrics[key] for best in history_best]
        frame = plot_metrics_comparison(key, history_metrics, n_iter, **self.visual_config)
        return frame

    def format_log_line(self) -> str:
        """格式化当前迭代日志行"""
        # 检查是否存在追踪指标（取交集后）
        if len(self.metrics_set):
            # 若存在追踪指标则默认使用第一个指标追踪
            prefix = f"[{self.metrics_set[0]}] "
        # 若不存在追踪指标则使用默认信息
        elif self.problem.n_objs == 1:
            # 单目标优化使用默认信息为目标值
            prefix = "[Obj] "
        else:
            # 多目标优化使用默认信息为前沿解数量
            warn_once(
                "No tracking metrics configured. Falling back to Pareto front size, "
                "which may not provide meaningful algorithm comparison results. "
                "Please use track_metrics() to specify metrics (e.g., track_metrics('hv')) "
                "for proper evaluation."
            )
            prefix = "[Front] "
        lines = list()
        # 迭代次数信息
        lines.append(f"Iter: {self.n_iter:0{len(str(self.max_iter))}d}/{self.max_iter}")
        for name, algorithm in self.algorithms.items():
            best = algorithm.get_best()
            # 检查是否存在追踪指标（取交集后）
            if len(self.metrics_set):
                # 若存在追踪指标则默认使用第一个指标追踪
                lines.append(f"{name}: {best.get_metric(self.metrics_set[0]):.6e}")
            # 单目标优化使用默认信息为目标值
            elif self.problem.n_objs == 1:
                obj_value = float(best.f) if best.f is not None else float('nan')
                lines.append(f"{name}: {obj_value:.6e}")
            # 多目标优化使用默认信息为前沿解数量
            else:
                lines.append(f"{name}: {len(best):{len(str(best.n_sols))}d}")
        return prefix + " | ".join(lines)

    def get_result_data(self) -> Dict:
        """
        获取多个算法优化结果数据
        :return: 按列索引的表格
        """
        # 初始化数据字典
        result_data = dict()
        # 初始化首列数据
        result_data['Algorithm'] = []
        # 如果是单目标，添加目标值与约束值列
        if self.problem.n_objs == 1:
            # 添加目标值
            result_data['Obj'] = []
            # 如果有约束，添加约束列
            if self.problem.n_cons > 0:
                result_data['Con'] = []
        # 如果是多目标，且没有追踪指标，则添加前沿解数量列
        if (self.problem.n_objs > 1 and
                len(self.metrics_set) == 0):
            result_data['Front'] = []
        # 添加指定的追踪指标
        for key in self.metrics_set:
            result_data[key] = []
        # 添加运行时间指标
        result_data['Time(s)'] = []
        # 填充数据（每行是一个算法）
        for name, algorithm in self.algorithms.items():
            # 算法名称
            result_data['Algorithm'].append(name)
            # 填充每一个指标数据
            for key in list(result_data.keys())[1:]:
                result_data[key].append(algorithm.get_metric_value(key))
        return result_data

    def report_result(self,
                      float_format: str = ".4e",
                      transpose: bool = True,
                      csv_path: Optional[str] = None):
        """
        报告算法优化结果信息，并可选保存到CSV文件
        :param float_format: 浮点数的格式化格式，默认为 ".6e"
        :param transpose: 是否将行与列转置（即将整个表格进行转置）
        :param csv_path: CSV文件保存路径，如果为None则只打印表格，不保存文件
        """
        result_data = self.get_result_data()
        # 使用按列索引的表格
        table_info = format_as_table(
            data=result_data,
            row_key='Algorithm',
            col_key='Metric',
            transpose=transpose,
            float_format=float_format
        )
        print(table_info)
        if csv_path:
            save_as_table(
                data=result_data,
                csv_path=csv_path,
                row_key='Algorithm',
                col_key='Metric',
                transpose=transpose,
                float_format=float_format
            )

    def set_visual_config(self, **kwargs):
        """更新绘图配置"""
        self.visual_config.update(kwargs)
        # 若与动画器参数相关则重新配置动画器参数
        self.animator.set_figsize(self.visual_config.get('fig_size', (6.4, 4.8)))
        self.animator.set_interval(self.visual_config.get('interval', 30))
        self.animator.set_save_frames(self.visual_config.get('save_frames', True))
