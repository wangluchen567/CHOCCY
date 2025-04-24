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
import copy
import matplotlib
import numpy as np
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
from typing import Union
from Algorithms import ALGORITHM
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor


class Evaluator(object):

    def __init__(self,
                 problems: Union[list, dict],
                 algorithms: Union[list, dict],
                 num_run: int = 5,
                 pop_size: Union[int, None] = None,
                 max_iter: Union[int, None] = None,
                 same_init: bool = False,
                 score_types: Union[str, dict, None] = None,
                 show_colors: Union[list, None] = None):
        """
        算法评估器(多问题多算法评估)
        :param problems: 问题集合(字典或列表)
        :param algorithms: 算法集合(字典或列表)
        :param num_run: 每种算法的运行次数
        :param pop_size: 每种算法初始化的种群大小
        :param max_iter: 每种算法的迭代次数
        :param same_init: 所有算法是否初始化相同
        :param score_types: 每个问题的评价指标分数
        :param show_colors: 指定每种算法的展示颜色
        """
        # 初始化给定参数
        self.num_run = num_run
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.same_init = same_init
        self.show_colors = show_colors
        self.problems = self._format(problems)
        self.algorithms = self._format(algorithms)
        # 初始化要使用的参数
        self.pairs = dict()  # 初始化'问题-算法'对
        self.scores = dict()  # 初始化'问题-算法'结果对
        self.test_marks = dict()  # 初始化'问题-算法'统计检验结果对
        # 初始化评价分数类型(支持每个问题独立指定不同类型)
        self.score_types = None
        self.set_score_types(score_types)
        # 指定绘图颜色(名称或HEX)
        self.colors = self.get_colors() if show_colors is None else show_colors
        # 若没有指定种群大小则默认使用算法集合中第一个算法的种群大小(可能会有bug)
        self.pop_size = next(iter(self.algorithms.values())).pop_size if self.pop_size is None else self.pop_size

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
                alg_temp = ALGORITHM(self.pop_size)
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

    def run_parallel(self, num_processes=None):
        """
        使用多进程并行运行评估任务(请谨慎使用,可能会造成卡顿)
        :param num_processes: 使用并行运行的CPU个数(默认使用CPU核心数)
        :return: None
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()  # 默认使用 CPU 核心数
            print(f'Using {num_processes} cpu parallel evaluate...')
        self.init_evaluator()
        # 逐个问题求解
        for (problem_name, pair) in self.pairs.items():
            for (alg_name, alg_repeats) in pair.items():
                print(f"{alg_name} solve {problem_name}...")

                # 使用进程池加速
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = [executor.submit(self.worker, alg_repeats[i]) for i in range(self.num_run)]
                    for i, future in enumerate(futures):
                        alg_repeats[i] = future.result()  # 获取运行后的对象

    def prints(self, score_types=None, stats_test=False, dec=6):
        """
        打印最终结果
        :param score_types: 指定每种问题的分数类型
        :param stats_test: 是否使用统计检验进行算法比较
        :param dec: 精确到小数点后的位数
        :return: None
        """
        if score_types is not None:
            # 若指定打印分数类型则重新对类型设置
            self.set_score_types(score_types)
        # 计算并记录每种算法指定类型的所有分数值
        self.record_scores()
        # 若使用统计检验比较算法
        if stats_test:
            self.compare_with_test()
        # 初始化每列的宽度
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
                alg_scores = self.scores[problem_name][alg_name]
                print_res = f"{alg_scores.mean():.{dec}e}({alg_scores.var():.{dec}e})"
                if stats_test:
                    print_res += self.test_marks[problem_name][alg_name]
                score_list.append(print_res)
                col_widths[idx + 1] = max(col_widths[idx + 1], len(score_list[idx + 1]) + 3)
            score_lists.append(score_list)
        # 打印结果
        titles_format = " ".join(f"{t:<{w}}" for t, w in zip(title_list, col_widths))
        print(titles_format)
        for score_list in score_lists:
            scores_format = " ".join(f"{s:<{w}}" for s, w in zip(score_list, col_widths))
            print(scores_format)

    def record_scores(self):
        """计算并记录每种算法指定类型的所有分数值"""
        for (problem_name, pair) in self.pairs.items():
            self.scores[problem_name] = dict()
            for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
                self.scores[problem_name][alg_name] \
                    = self.get_scores(alg_repeats, self.score_types[problem_name])

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

    def compare_with_test(self):
        """使用统计检验来比较算法优劣（与最后一个算法相比）"""
        for (problem_name, pair) in self.pairs.items():
            self.test_marks[problem_name] = dict()
            # 取最后一个算法的分数
            last_name = list(self.scores[problem_name].keys())[-1]
            last_scores = list(self.scores[problem_name].values())[-1]
            for idx, (alg_name, alg_repeats) in enumerate(pair.items()):
                if last_name == alg_name:
                    # 每个算法不与自己进行比较
                    self.test_marks[problem_name][alg_name] = ''
                else:
                    # 进行统计检验比较最后一个算法与当前算法的优劣
                    self.test_marks[problem_name][alg_name] \
                        = self.stats_test_scores(last_scores,
                                                 self.scores[problem_name][alg_name],
                                                 self.score_types[problem_name])

    @staticmethod
    def stats_test_scores(scores1, scores2, score_type, alpha=0.05):
        """
        使用秩和检验比较两组分数值的优劣
        :param scores1: 分数组1
        :param scores2: 分数组2
        :param score_type: 分数类型
        :param alpha: 显著性水平，默认0.05
        :return: 优劣比较
        """
        # 返回的是相较于组1，组2的优劣情况
        res = ['-', '+', '=']
        if score_type.lower() in ['HV']:
            res = ['+', '-', '=']
        stats, p = mannwhitneyu(scores1, scores2, alternative='two-sided')
        if p < alpha:
            # 显著性水平小于alpha则说明两组数据存在明显差异
            if stats < len(scores1) * len(scores2) / 2:
                return res[0]
            else:
                return res[1]
        else:
            # 否则两者水平不存在差异
            return res[2]

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

    def plot_violin(self, problem_name=None, default_color=True, cut=2):
        """绘制指定问题的小提琴图以对比算法(cut=0时准确绘制)"""
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
            sns.violinplot(data=datas, inner="quartile", cut=cut)
        else:
            sns.violinplot(data=datas, inner="quartile", cut=cut, palette=self.colors)
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
        plt.xlabel(self.score_types[problem_name])
        plt.ylabel('Density')
        plt.grid()
        plt.legend()
        # 显示图形
        plt.show()

    @staticmethod
    def worker(alg):
        """工作器"""
        alg.run()
        return alg
