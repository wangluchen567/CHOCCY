import copy
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
from typing import Union
from Algorithms.ALGORITHM import ALGORITHM


class Evaluator(object):

    def __init__(self,
                 problems: Union[list, dict],
                 algorithms: Union[list, dict],
                 num_run: int = 1,
                 num_pop: Union[int, None] = None,
                 num_iter: Union[int, None] = None,
                 same_init: bool = False,
                 score_types: Union[str, dict, None] = None,
                 show_colors: Union[list, None] = None):
        """算法评估器(多问题多算法评估)"""

        self.num_run = num_run
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.same_init = same_init
        self.show_colors = show_colors
        self.problems = self._format(problems)
        self.algorithms = self._format(algorithms)

        self.pairs = dict()  # 初始化'问题-算法'对
        # 初始化评价分数类型(支持每个问题独立指定不同类型)
        self.score_types = None
        self.set_score_types(score_types)
        # 指定绘图颜色(名称或HEX)
        self.colors = self.get_colors() if show_colors is None else show_colors
        # 若没有指定种群大小则默认使用算法集合中第一个算法的种群大小(可能会有bug)
        self.num_pop = next(iter(self.algorithms.values())).num_pop if self.num_pop is None else self.num_pop

    @staticmethod
    def _format(value):
        """格式化列表(将列表变为字典)"""
        if isinstance(value, dict):
            return value
        elif isinstance(value, list):
            return {type(item).__name__: item for item in value}
        else:
            raise ValueError("This type of conversion is not supported")

    def init_evaluator(self):
        """初始化评估器"""
        self.pairs = dict()  # 初始化'问题-算法'对
        # 根据所有问题对所有算法进行初始化
        for (problem_name, problem) in self.problems.items():
            # 初始化问题对应算法字典
            self.pairs[problem_name] = dict()
            pop = None  # 初始化种群
            if self.same_init:
                # 若需要相同初始化则
                # 创建一个算法对象用于初始化种群
                alg_temp = ALGORITHM(self.num_pop)
                alg_temp.init_algorithm(problem)
                pop = alg_temp.pop  # 得到种群
            for (alg_name, alg) in self.algorithms.items():
                # 不修改算法原来参数
                alg_ = copy.deepcopy(alg)
                alg_.show_mode = -1
                if self.same_init:
                    alg_.init_algorithm(problem, pop.copy())
                else:
                    alg_.init_algorithm(problem)
                # 将算法复制多次以运行num_run次
                self.pairs[problem_name][alg_name] = [copy.deepcopy(alg_) for _ in range(self.num_run)]

    def set_score_types(self, score_types):
        """初始化评价分数类型"""
        if score_types is None:
            # 若指定类型为空，则使用默认指定
            self.score_types = dict()
            for (problem_name, problem) in self.problems.items():
                if problem.num_obj == 1:
                    self.score_types[problem_name] = "Fitness"
                else:
                    self.score_types[problem_name] = "HV"
        elif isinstance(score_types, str):
            # 若只指定一种类型则所有问题均指定为该类型
            self.score_types = dict()
            for (problem_name, problem) in self.problems.items():
                self.score_types[problem_name] = score_types
        elif isinstance(score_types, list):
            assert len(score_types) == len(self.problems)
            self.score_types = dict()
            for idx, (problem_name, problem) in enumerate(self.problems.items()):
                self.score_types[problem_name] = score_types[idx]
        elif isinstance(score_types, dict):
            self.score_types = score_types
        else:
            raise ValueError("This type of conversion is not supported")

    def run(self):
        """运行评估器"""
        self.init_evaluator()
        # 逐个问题求解
        for (problem_name, pair) in self.pairs.items():
            for (alg_name, alg_repeats) in pair.items():
                print(f"{alg_name} solve {problem_name}...")
                for i in tqdm(range(self.num_run)):
                    alg_repeats[i].run()

    def prints(self, score_types=None, dec=6):
        """打印最终结果"""
        if score_types is not None:
            # 若指定打印分数类型则重新对类型设置
            self.set_score_types(score_types)
        col_widths = np.zeros(len(self.algorithms) + 1, dtype=int)
        # 第一行是算法的名称
        title_list = ["Algorithm"]
        for idx, alg_name in enumerate(self.algorithms.keys()):
            title_list.append(alg_name)
            col_widths[idx] = max(col_widths[idx], len(alg_name) + 3)
        # 每一行都是问题与分数结果
        score_lists = []
        for (problem_name, pair) in self.pairs.items():
            score_list = [problem_name]
            col_widths[0] = max(col_widths[0], len(problem_name) + 3)
            for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
                scores = self.get_scores(alg_repeats, self.score_types[problem_name])
                score_list.append(f"{scores.mean():.{dec}e}({scores.var():.{dec}e})")
                col_widths[idx + 1] = max(col_widths[idx + 1], len(score_list[idx + 1]) + 3)
            score_lists.append(score_list)
        # 打印结果
        titles_format = " ".join(f"{t:<{w}}" for t, w in zip(title_list, col_widths))
        print(titles_format)
        for score_list in score_lists:
            scores_format = " ".join(f"{s:<{w}}" for s, w in zip(score_list, col_widths))
            print(scores_format)

    @staticmethod
    def get_scores(alg_repeats, score_type):
        """给定算法集合求算法集合的指定类型分数"""
        scores = np.zeros(len(alg_repeats))
        if score_type.lower() == 'time':
            # 若是要求显示时间则统计时间
            for i in range(len(alg_repeats)):
                scores[i] = alg_repeats[i].run_time
        elif score_type.lower() == 'fitness':
            # 单目标问题则直接为最优目标值
            for i in range(len(alg_repeats)):
                scores[i] = alg_repeats[i].best_obj
        else:
            # 多目标问题则指定分数值
            for i in range(len(alg_repeats)):
                scores[i] = alg_repeats[i].cal_score(score_type, best_obj=alg_repeats[i].best_obj)
        return scores

    def get_colors(self):
        """绘图颜色设置"""
        num_colors = len(self.algorithms)
        if num_colors <= 3:
            # 若数量少则直接指定颜色
            colors = ['blue', 'red', 'green']
            return colors
        # 否则从彩虹色图中采样颜色
        # 检查 Matplotlib 版本
        if matplotlib.__version__ >= '3.7':
            rainbow_cmap = plt.colormaps['rainbow']
        else:
            rainbow_cmap = plt.cm.get_cmap('rainbow')
        # 生成 num_colors 种颜色
        raw_colors = rainbow_cmap(np.linspace(0, 1, num_colors))
        # 将颜色转换为十六进制格式
        colors = [mcolors.to_hex(c) for c in raw_colors]
        return colors

    def plot_violin(self, problem_name=None, default_color=True):
        """绘制指定问题的小提琴图以对比算法"""
        plt.figure()
        if problem_name is None:
            # 若指定问题为空则默认第一个问题
            problem_name = next(iter(self.pairs.keys()))
        pair = self.pairs[problem_name]
        ticks, datas, labels = [], [], []
        for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
            scores = self.get_scores(alg_repeats, self.score_types[problem_name])
            ticks.append(idx)
            datas.append(scores)
            labels.append(alg_name)
        # 创建小提琴图
        if default_color:
            sns.violinplot(data=datas, inner="quartile")
        else:
            sns.violinplot(data=datas, inner="quartile", palette=self.colors)
        # 添加标题和标签
        plt.title(problem_name)
        plt.xlabel('Algorithms')
        plt.ylabel(self.score_types[problem_name])
        plt.xticks(ticks, labels)
        # 显示图形
        plt.show()

    def plot_box(self, problem_name=None, default_color=True):
        """绘制指定问题箱线图以对比算法"""
        plt.figure()
        if problem_name is None:
            # 若指定问题为空则默认第一个问题
            problem_name = next(iter(self.pairs.keys()))
        pair = self.pairs[problem_name]
        ticks, datas, labels = [], [], []
        for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
            scores = self.get_scores(alg_repeats, self.score_types[problem_name])
            ticks.append(idx)
            datas.append(scores)
            labels.append(alg_name)
        # 创建箱线图
        if default_color:
            sns.boxplot(data=datas)
        else:
            sns.boxplot(data=datas, palette=self.colors)
        # 添加标题和标签
        plt.title(problem_name)
        plt.xlabel('Algorithms')
        plt.ylabel(self.score_types[problem_name])
        plt.xticks(ticks, labels)
        # 显示图形
        plt.show()

    def plot_kde(self, problem_name=None, default_color=True):
        """绘制指定问题核密度估计图以对比算法"""
        plt.figure()
        if problem_name is None:
            # 若指定问题为空则默认第一个问题
            problem_name = next(iter(self.pairs.keys()))
        pair = self.pairs[problem_name]
        ticks, datas, labels = [], [], []
        for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
            scores = self.get_scores(alg_repeats, self.score_types[problem_name])
            ticks.append(idx)
            datas.append(scores)
            labels.append(alg_name)
            # 创建核密度估计图
            if default_color:
                sns.kdeplot(data=scores, label=alg_name, fill=True)
            else:
                sns.kdeplot(data=scores, label=alg_name, fill=True, color=self.colors[idx])
        # 添加标题和标签
        plt.title(problem_name)
        plt.xlabel('Algorithms')
        plt.ylabel(self.score_types[problem_name])
        plt.grid()
        plt.legend()
        # 显示图形
        plt.show()
