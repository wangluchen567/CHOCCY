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
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from typing import Union
from Problems import PROBLEM
from Algorithms import View
from Algorithms.Utility.RecordUtils import setup_logger
from Algorithms.Utility.ReadUtils import load_array, load_arrays
from Algorithms.Utility.SaveUtils import save_array, save_arrays, get_timestamp, save_json
from Algorithms.Utility.PlotUtils import plot_scores, plot_decs, plot_objs, plot_objs_decs
from Algorithms.Utility.SupportUtils import fast_nd_sort, shuffle_matrix_in_row, record_time
from Algorithms.Utility.PerfMetrics import cal_gd, cal_igd, cal_gd_plus, cal_igd_plus, cal_hv
from Algorithms.Utility.Selections import elitist_selection, tournament_selection, roulette_selection
from Algorithms.Utility.Operators import operator_real, operator_binary, operator_permutation, operator_fix_label


class ALGORITHM(object):
    # 定义问题常量
    REAL = PROBLEM.REAL  # 实数
    INT = PROBLEM.INT  # 整数
    BIN = PROBLEM.BIN  # 二进制
    PMU = PROBLEM.PMU  # 序列
    FIX = PROBLEM.FIX  # 固定标签
    # 定义绘图常量
    NONE = View.NONE  # 不绘制
    BAR = View.BAR  # 绘制进度条
    OBJ = View.OBJ  # 绘制目标空间
    DEC = View.DEC  # 绘制决策空间
    MIX2D = View.MIX2D  # 绘制目标空间和决策空间混合(二维空间)
    MIX3D = View.MIX3D  # 绘制目标空间和决策空间混合(三维空间)
    SCORE = View.SCORE  # 绘制分数情况(单目标为目标值,多目标为评价指标)
    PROB = View.PROB  # 问题提供绘图方法
    ALGO = View.ALGO  # 算法提供绘图方法
    LOG = View.LOG  # 输出日志
    # 定义指标类型常量(多目标)
    score_types = ['HV', 'GD', 'IGD', 'GD+', 'IGD+']

    def __init__(self,
                 pop_size: Union[int, None] = None,
                 max_iter: Union[int, None] = None,
                 cross_prob: Union[float, None] = None,
                 mutate_prob: Union[float, None] = None,
                 educate_prob: Union[float, None] = None,
                 show_mode: Union[int, View] = BAR):
        """
        算法父类

        Code Author: LuChen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param educate_prob: 教育概率
        :param show_mode: 绘图模式
        """
        # 初始化给定参数
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.show_mode = show_mode
        # 初始化交叉、变异和教育概率
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.educate_prob = educate_prob
        # 初始化问题上下界
        self.lower, self.upper = None, None
        # 初始化问题决策向量与目标向量大小
        self.num_dec, self.num_obj = None, None
        # 初始化问题对象与问题类型
        self.problem, self.problem_type = None, None
        # 初始化问题类型情况和每个问题类别对应的位置
        self.unique_type, self.type_indices = None, None
        # 算法是否只能求解单目标问题(默认可全部求解)
        self.only_solve_single = False
        # 算法可求解的问题类型(默认可全部求解)
        self.solvable_type = [self.REAL, self.INT, self.BIN, self.PMU, self.FIX]
        # 初始化种群解及其目标/约束/适应度
        self.pop, self.objs, self.cons, self.fits = None, None, None, None
        # 记录种群解及其目标/约束(适应度为中间值不记录)
        self.pop_history, self.objs_history, self.cons_history = [], [], []
        # 初始化种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best, self.best_obj, self.best_con = None, None, None
        # 记录种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best_history, self.best_obj_his, self.best_con_his = [], [], []
        # 记录开始时间
        self.start_time = 0.0
        # 记录运行时间
        self.run_time = 0.0
        # 初始化日志记录器
        self.logger = None
        # 初始化决策变量示例
        self.example_dec = None
        # 初始化评价指标记录与评价指标类型
        self.scores, self.score_type = np.empty(0), None

    @record_time
    def init_algorithm(self, problem: PROBLEM, pop=None):
        """
        初始化算法
        :param problem: 问题对象
        :param pop: 指定种群用于初始化
        """
        # 初始化算法所有参数
        self.init_params(problem)
        # 检查算法是否可求解该问题
        self.check_feasibility()
        # 初始化种群并对种群进行评价
        self.init_and_eval(pop)

    def init_params(self, problem: PROBLEM):
        """
        初始化所有参数
        :param problem: 问题对象
        """
        # 初始化问题参数
        self.problem = problem
        self.num_dec = self.problem.num_dec
        self.num_obj = self.problem.num_obj
        self.problem_type = self.problem.problem_type
        self.unique_type = self.problem.unique_type
        self.type_indices = self.problem.type_indices
        self.lower = self.problem.lower
        self.upper = self.problem.upper
        self.start_time = time.time()
        # 决策变量示例(固定标签问题)
        if hasattr(self.problem, 'example_dec'):
            self.example_dec = self.problem.example_dec
        # 初始化评价指标类型(单目标为适应度, 多目标默认为超体积指标)
        self.score_type = 'Fitness' if self.num_obj == 1 else 'HV'
        # 初始化算法参数
        self.pop_size = 100 if self.pop_size is None else self.pop_size
        self.max_iter = 100 if self.max_iter is None else self.max_iter
        self.cross_prob = 1.0 if self.cross_prob is None else self.cross_prob
        self.educate_prob = 0.5 if self.educate_prob is None else self.educate_prob
        self.mutate_prob = 1 / self.num_dec if self.mutate_prob is None else self.mutate_prob
        self.logger = setup_logger() if self.show_mode == self.LOG else None  # 设置日志记录器

    def check_feasibility(self):
        """检查算法是否可求解该问题"""
        if self.only_solve_single and self.num_obj > 1:
            raise ValueError("This algorithm can only solve single objective problems")
        if not np.all(np.isin(self.unique_type, self.solvable_type)):
            raise ValueError("This algorithm does not support solving this type of problem")

    def init_and_eval(self, pop=None):
        """
        初始化种群并对种群中解进行评价
        :param pop: 指定种群用于初始化
        """
        # 若给定种群中个体数量太多则进行裁剪
        pop = pop[:self.pop_size] if pop is not None else None
        # 初始化种群(随机初始化或给定种群初始化)
        self.pop = self.init_pop() if pop is None else pop
        # 对种群中解进行评价(求目标值/约束值/适应度)
        self.objs, self.cons, self.fits = self.evaluate(self.pop)
        # 记录当前种群信息
        self.record()

    def get_iterator(self):
        """构建迭代器"""
        if self.show_mode == 0:
            return tqdm(range(self.max_iter))
        else:
            return range(self.max_iter)

    def solve(self, problem: PROBLEM):
        """算法求解问题(主入口函数)"""
        # 初始化算法
        self.init_algorithm(problem)
        # 运行算法求解问题
        self.run()

    def run(self):
        """运行算法"""
        # 绘制初始状态图
        self.plot(n_iter=0, pause=True)
        # 算法迭代并优化问题
        for i in self.get_iterator():
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(n_iter=i + 1, pause=True)

    def run_step(self, *args, **kwargs):
        """运行算法单步"""
        self.record()

    def cal_objs(self, pop):
        """
        计算目标值
        :param pop: 给定种群
        :return: 种群中每个个体目标值（矩阵）
        """
        return self.problem.cal_objs(pop)

    def cal_cons(self, pop):
        """
        计算约束值
        :param pop: 给定种群
        :return: 种群中每个个体约束值（矩阵）
        """
        return self.problem.cal_cons(pop)

    @staticmethod
    def cal_objs_based_cons(objs, cons):
        """
        计算约束松弛后的目标值
        :param objs: 种群中每个个体目标值（矩阵）
        :param cons: 种群中每个个体约束值（矩阵）
        :return: 基于约束松弛后的每个个体目标值（矩阵）
        """
        objs_based_cons = objs.copy()
        # 找出所有不满足约束的个体
        not_feas = np.any(cons > 0, axis=1)
        # 计算当前种群中每个目标的最大值
        max_objs = np.max(objs, axis=0)
        # 计算不满足约束的个体的不满足约束的程度值
        penalty = np.sum(np.maximum(cons[not_feas], 0), axis=1)
        # 利用广播机制更新不满足约束的个体的目标函数值
        objs_based_cons[not_feas] = max_objs + penalty.reshape(-1, 1)

        return objs_based_cons

    def cal_fits(self, objs, cons):
        """
        根据给定目标值和约束值得到适应度值(默认是单目标情况)
        :param objs: 种群中每个个体目标值（矩阵）
        :param cons: 种群中每个个体约束值（矩阵）
        :return: 种群中每个个体适应度值（矩阵）
        """
        # 检查是否均满足约束，若均满足约束则无需考虑约束
        if np.all(cons <= 0):
            return objs.flatten()
        else:
            return self.cal_objs_based_cons(objs, cons).flatten()

    def evaluate(self, pop):
        """
        给定种群解并对解进行评价(求目标值/约束值/适应度)
        :param pop: 给定种群
        :return: 目标值/约束值/适应度
        """
        objs = self.cal_objs(pop)
        cons = self.cal_cons(pop)
        fits = self.cal_fits(objs, cons)
        return objs, cons, fits

    def set_score_type(self, score_type):
        """
        重新设置评价指标分数类型(多目标)
        :param score_type: 评价指标(分数)类型
        """
        if self.num_obj == 1:
            warnings.warn("Single objective problem cannot set score type")
            return
        if score_type not in self.score_types:
            raise ValueError(f"There is no {score_type} score type")
        self.score_type = score_type

    def cal_score(self, score_type=None, best_obj=None, optimums=None):
        """
        给定种群解的最优个体目标值计算评价指标分数(多目标)
        :param score_type: 评价指标分数类型
        :param best_obj: 最优目标值情况
        :param optimums: 最优解（或参考点）
        :return: 评价指标分数(多目标)
        """
        if score_type is None:
            score_type = self.score_type
        if best_obj is None:
            best_obj = self.best_obj
        if optimums is None:
            optimums = self.problem.optimums
        if score_type == 'HV':
            return cal_hv(best_obj, optimums)
        elif score_type == 'GD':
            return cal_gd(best_obj, optimums)
        elif score_type == 'IGD':
            return cal_igd(best_obj, optimums)
        elif score_type == 'GD+':
            return cal_gd_plus(best_obj, optimums)
        elif score_type == 'IGD+':
            return cal_igd_plus(best_obj, optimums)
        else:
            raise ValueError(f"There is no {self.score_type} score type")

    @staticmethod
    def record_time(method):
        """
        统计运行时间
        :param method: 方法
        :return: 运行时间
        """
        return record_time(method)

    def get_best(self):
        """获取种群最优解与其目标值约束值"""
        if self.num_obj == 1:
            return self.best, self.best_obj[0], self.best_con[0]
        else:
            return self.best, self.best_obj, self.best_con

    def init_pop(self):
        """初始化种群"""
        init_dict = {ALGORITHM.REAL: self.init_pop_real,
                     ALGORITHM.INT: self.init_pop_integer,
                     ALGORITHM.BIN: self.init_pop_binary,
                     ALGORITHM.PMU: self.init_pop_permutation,
                     ALGORITHM.FIX: self.init_pop_fix_label}
        pop = np.zeros((self.pop_size, self.num_dec))
        # 若没有实数或整数部分则直接初始化为整型
        if np.all(self.unique_type > 1):
            pop = np.zeros((self.pop_size, self.num_dec), dtype=int)
        # 遍历所有问题类型
        for t in self.unique_type:
            pop[:, self.type_indices[t]] = init_dict.get(t)()
        return pop

    def init_pop_real(self):
        """初始化求解实数或整数问题的种群"""
        pop = np.random.uniform(self.lower[self.type_indices[ALGORITHM.REAL]],
                                self.upper[self.type_indices[ALGORITHM.REAL]],
                                size=(self.pop_size, len(self.type_indices[ALGORITHM.REAL])))
        return pop

    def init_pop_integer(self):
        """初始化求解实数或整数问题的种群"""
        pop = np.random.uniform(self.lower[self.type_indices[ALGORITHM.INT]],
                                self.upper[self.type_indices[ALGORITHM.INT]],
                                size=(self.pop_size, len(self.type_indices[ALGORITHM.INT])))
        return pop

    def init_pop_binary(self):
        """初始化求解二进制问题的种群"""
        pop = np.random.randint(2, size=(self.pop_size, len(self.type_indices[ALGORITHM.BIN])))
        return pop

    def init_pop_permutation(self):
        """初始化求解序列问题的种群"""
        pop = np.argsort(np.random.uniform(0, 1,
                                           size=(self.pop_size, len(self.type_indices[ALGORITHM.PMU]))), axis=1)
        return pop

    def init_pop_fix_label(self):
        """初始化求解固定标签问题的种群"""
        # 确保给定的示例和决策向量大小相等
        if len(self.type_indices[ALGORITHM.FIX]) != len(self.example_dec):
            raise ValueError("The given example and decision vector are not of equal size")
        # 确定初始向量
        pop = self.example_dec.copy()
        # 初始化种群向量
        pop = np.repeat(pop.reshape(1, -1), self.pop_size, axis=0)
        # 打乱每行的个体向量
        shuffle_matrix_in_row(pop)
        # 确保为整型
        pop = np.array(pop, dtype=int)
        return pop

    def operator(self, mating_pool):
        """
        进行交叉变异生成子代
        :param mating_pool: 匹配池(下标)
        :return: 子代种群
        """
        offspring = self.pop[mating_pool]
        for t in self.unique_type:
            offspring[:, self.type_indices[t]] = self.operator_(t)(offspring[:, self.type_indices[t]],
                                                                   self.lower[self.type_indices[t]],
                                                                   self.upper[self.type_indices[t]],
                                                                   self.cross_prob, self.mutate_prob)
        return offspring

    @staticmethod
    def operator_(problem_type):
        """
        根据问题类型返回对应函数
        :param problem_type: 问题类型
        :return: 操作函数(交叉变异)
        """
        if problem_type == ALGORITHM.REAL:
            return operator_real
        elif problem_type == ALGORITHM.INT:
            return operator_real
        elif problem_type == ALGORITHM.BIN:
            return operator_binary
        elif problem_type == ALGORITHM.PMU:
            return operator_permutation
        elif problem_type == ALGORITHM.FIX:
            return operator_fix_label
        else:
            raise ValueError(f"The problem type {problem_type} does not exist")

    def educate(self, *args, **kwargs):
        """对子代进行教育"""
        pass

    def mating_pool_selection(self, num_next=None, k=2):
        """
        匹配池选择
        :param num_next: 下一代种群的个体数量
        :param k: 用于锦标赛选择，K元锦标赛
        :return: 匹配池（下标）
        """
        if num_next is None:
            num_next = self.pop_size
        if k >= 2:
            # 使用锦标赛选择获取匹配池
            return tournament_selection(self.fits, num_next, k)
        else:
            # 使用轮盘选择法获取匹配池
            return roulette_selection(self.fits, num_next)

    def pop_merge(self, offspring):
        """
        当前种群与其子代合并
        :param offspring:
        :return: 新种群及其目标值/约束值/适应度值
        """
        # 先计算子代目标值与约束值
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 将父代与子代合并获得新种群
        new_pop = np.vstack((self.pop, offspring))
        new_objs = np.vstack((self.objs, off_objs))
        new_cons = np.vstack((self.cons, off_cons))
        # 重新计算合并种群的的等价适应度值
        new_fits = self.cal_fits(new_objs, new_cons)
        return new_pop, new_objs, new_cons, new_fits

    def environmental_selection(self, offspring):
        """
        进行环境选择
        :param offspring: 子代种群
        """
        # 将当前种群与其子代合并
        new_pop, new_objs, new_cons, new_fits = self.pop_merge(offspring)
        # 使用选择策略(默认精英选择)选择进入下一代新种群的个体
        best_indices = elitist_selection(new_fits, self.pop_size)
        self.pop = new_pop[best_indices]
        self.objs = new_objs[best_indices]
        self.cons = new_cons[best_indices]
        self.fits = new_fits[best_indices]

    def get_current_best(self):
        """获取当前种群的最优解"""
        # 若有整数问题则需要对种群进行向下取整
        if self.INT in self.type_indices:
            current_pop = self.pop.copy()
            current_pop[:, self.type_indices[self.INT]] = np.floor(self.pop[:, self.type_indices[self.INT]])
            self.best, self.best_obj, self.best_con = self.get_current_best_(current_pop, self.objs, self.cons)
        else:
            self.best, self.best_obj, self.best_con = self.get_current_best_(self.pop, self.objs, self.cons)

    @staticmethod
    def get_current_best_(pop, objs, cons):
        """
        获取给定种群的最优解
        :param pop: 给定种群
        :param objs: 给定种群目标值
        :param cons: 给定种群约束值
        :return: 给定种群最优解/解集
        """
        num_obj = objs.shape[1]
        # 先判断是否满足约束
        feas = (cons <= 0).flatten()
        pop_sat = pop[feas]
        objs_sat = objs[feas]
        cons_sat = cons[feas]
        if len(pop_sat) == 0:
            # 若没有满足约束的解则返回约束最小的解
            min_index = np.argmin(cons)
            best = pop[min_index]
            best_obj = objs[min_index]
            best_con = cons[min_index]
            return best, best_obj, best_con
        elif num_obj == 1:
            # 若目标个数为1，则只选择一个最优解返回
            min_index = np.argmin(objs_sat)
        else:
            # 若为多目标问题，则返回全部的最优前沿
            fronts, _ = fast_nd_sort(objs_sat)
            min_index = fronts[0]
        best = pop_sat[min_index]
        best_obj = objs_sat[min_index]
        best_con = cons_sat[min_index]
        return best, best_obj, best_con

    def record(self):
        """记录当前种群的信息"""
        # 记录种群个体及其目标值
        self.pop_history.append(self.pop.copy())
        self.objs_history.append(self.objs.copy())
        self.cons_history.append(self.cons.copy())
        # 记录种群最优个体及其目标和约束值
        self.get_current_best()
        self.best_history.append(self.best)
        self.best_obj_his.append(self.best_obj)
        self.best_con_his.append(self.best_con)
        if self.num_obj == 1 or self.show_mode == self.SCORE or self.show_mode == self.LOG:
            # 若是单目标问题或需要展示评价指标变化则直接记录分数
            self.record_score()

    def record_score(self):
        """记录分数值"""
        if self.num_obj == 1:
            self.scores = self.best_obj_his
        else:
            score_value = self.cal_score()
            self.scores = np.append(self.scores, score_value)

    def clear_record(self):
        """清除所有记录"""
        # 记录种群解及其目标/约束(适应度为中间值不记录)
        self.pop_history, self.objs_history, self.cons_history = [], [], []
        # 初始化种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best, self.best_obj, self.best_con = None, np.inf, np.inf
        # 记录种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best_history, self.best_obj_his, self.best_con_his = [], [], []
        # 记录评价指标
        self.scores = np.empty(0)

    def plot(self, show_mode=None, n_iter=None, pause=False, sym=True):
        """
        绘图函数，根据不同模式进行绘图
        :param show_mode: 绘图模式，参见View类
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param sym: 是否进行完全对称图像的绘制（只对MIX绘图有效）
        """
        if n_iter == self.max_iter:
            # 最后一次迭代不再使用停顿展示
            pause = False
        if show_mode is not None:
            self.show_mode = show_mode
        if self.show_mode == self.NONE or self.show_mode == self.BAR:
            pass
        elif self.show_mode == self.LOG:
            self.logger.info(self.get_log_info(n_iter))
        elif self.show_mode == self.OBJ:
            self.plot_objs(n_iter, pause)
        elif self.show_mode == self.DEC:
            self.plot_decs(n_iter, pause)
        elif self.show_mode == self.MIX2D:
            self.plot_objs_decs(n_iter, pause, sym=sym)
        elif self.show_mode == self.MIX3D:
            self.plot_objs_decs(n_iter, pause, sym=sym, contour=False)
        elif self.show_mode == self.SCORE:
            self.plot_scores(n_iter, pause)
        elif self.show_mode == self.PROB:
            self.plot_by_problem(n_iter, pause)
        elif self.show_mode == self.ALGO:
            self.plot_(n_iter, pause)
        else:
            raise ValueError(f"There is no such plotting mode: {self.show_mode}")

    def plot_(self, *args, **kwargs):
        """提供算法自定义绘图的接口"""
        pass

    def plot_decs(self, n_iter=None, pause=False, pause_time=0.06):
        """
        绘制种群个体决策向量
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        """
        if pause or n_iter is None:
            plot_decs(self.pop, n_iter, pause, pause_time)
        else:
            plot_decs(self.pop_history[n_iter], n_iter, pause, pause_time)

    def plot_objs(self, n_iter=None, pause=False, pause_time=0.06):
        """
        绘制种群目标值
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        """
        if pause or n_iter is None:
            if self.num_obj == 1:
                # 若是单目标问题，绘制目标值范围情况
                plot_objs(self.objs_history, n_iter, pause, pause_time)
            else:
                plot_objs(self.objs, n_iter, pause, pause_time, self.problem.pareto_front)
        else:
            if self.num_obj == 1:
                # 若是单目标问题，绘制目标值范围情况
                plot_objs(self.objs_history[:n_iter + 1], n_iter, pause, pause_time)
            else:
                plot_objs(self.objs_history[n_iter], n_iter, pause, pause_time, self.problem.pareto_front)

    def plot_objs_decs(self, n_iter=None, pause=False, pause_time=0.06, contour=True, sym=True):
        """
        在特定条件下可将目标空间与决策空间绘制到同一空间中
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        :param contour: 是否使用等高线的方式绘制
        :param sym: 是否进行完全对称图像的绘制
        """
        if pause or n_iter is None:
            plot_objs_decs(self.problem, self.pop, self.objs,
                           n_iter, pause, pause_time, contour=contour, sym=sym)
        else:
            plot_objs_decs(self.problem, self.pop_history[n_iter], self.objs_history[n_iter],
                           n_iter, pause, pause_time, contour=contour, sym=sym)

    def get_scores(self):
        """获取历史所有种群的评价分数"""
        # 若之前已经记录评价分数则直接返回分数
        if self.scores is not None and len(self.scores):
            return self.scores
        # 若是单目标问题则评价分数就是最优目标值
        if self.num_obj == 1:
            self.scores = self.best_obj_his
            return self.scores
        # 若是多目标问题则计算评价分数
        self.scores = np.zeros(len(self.best_obj_his))
        for i in range(len(self.best_obj_his)):
            self.scores[i] = self.cal_score(best_obj=self.best_obj_his[i])
        return self.scores

    def plot_scores(self, n_iter=None, pause=False, pause_time=0.06):
        """
        绘制指标的变化情况
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        """
        if self.scores is None or len(self.scores) == 0:
            self.get_scores()
        if pause or n_iter is None:
            plot_scores(self.scores, self.score_type, n_iter, pause, pause_time)
        else:
            plot_scores(self.scores[:n_iter + 1], self.score_type, n_iter, pause, pause_time)

    def plot_by_problem(self, n_iter=None, pause=False):
        """
        使用问题给定的绘图函数绘图
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        """
        if pause or n_iter is None:
            self.problem.plot(self.best_history[-1], n_iter, pause)
        else:
            self.problem.plot(self.best_history[n_iter], n_iter, pause)

    def enable_file_logging(self, log_name=None):
        """
        打开输出日志到文件功能
        :param log_name: 输出日志文件的名称
        """
        if self.show_mode != self.LOG:
            warnings.warn("Logging mode is not currently enabled")
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        if log_name is None:
            # 获取算法名称
            algo_name = type(self).__name__
            # 若log_name给定为空则默认命名为[algorithm_name].log
            log_name = algo_name + ".log"
        save_path = project_root + "\\Outputs\\Logs\\" + log_name
        self.logger = setup_logger(save_path, to_file=True)

    def get_log_info(self, n_iter):
        """
        获取当前状态信息
        :param n_iter: 迭代次数
        """
        info = ""
        use_time = time.time() - self.start_time
        # 获取算法与问题名称
        algo_name = type(self).__name__
        prob_name = type(self.problem).__name__
        # 判断是否是单目标问题
        if self.num_obj == 1:
            info += f"[{algo_name}] n_iter = {n_iter}, time = {use_time:.3f} s, "
            info += f"[{prob_name}] best_obj = {self.best_obj[0]:.6e}"
        else:
            info += f"[{algo_name}] n_iter = {n_iter}, time = {use_time:.3f} s, "
            info += f"[{prob_name}] {self.score_type} = {self.scores[-1]:.6e}"
        if self.problem.overwrite_cons:
            # 如果问题有约束条件(覆写了约束方法)则输出约束情况
            if self.num_obj == 1:
                info += f", best_con = {self.best_con[0]:.6e}"
            info += f", feasible_ratio = {np.sum(self.cons <= 0)}/{self.pop_size}"
        return info

    def get_params_info(self):
        """获取算法的参数信息"""
        return {
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'cross_prob': self.cross_prob,
            'mutate_prob': self.mutate_prob,
            'educate_prob': self.educate_prob,
            'only_solve_single': self.only_solve_single
        }

    def save_best(self, save_type='csv'):
        """
        保存最优个体解的结果
        :param save_type: 保存文件类型
        """
        # 获取算法与问题名称
        algo_name = type(self).__name__
        prob_name = type(self.problem).__name__
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        # 得到当前时间点时间戳
        timestamp = get_timestamp()
        if self.problem.num_obj == 1:  # 判断是否是单目标问题
            save_path = project_root + "\\Outputs\\Single\\" + algo_name + "_solve_" + prob_name + "_best_" + timestamp
        else:
            save_path = project_root + "\\Outputs\\Multi\\" + algo_name + "_solve_" + prob_name + "_best_" + timestamp
        # 若文件夹不存在则创建文件夹
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存问题和算法的相关信息
        info = dict()
        info['algorithm_params'] = self.get_params_info()
        info['problem_info'] = self.problem.get_info()
        save_json(info, save_path + "\\info")
        # 保存相关数据
        try:
            save_array(self.best, save_path + "\\best", save_type)
            save_array(self.best_obj, save_path + "\\best_obj", save_type)
            save_array(self.best_con, save_path + "\\best_con", save_type)
        except Exception as e:
            warnings.warn(f"There is a error with saving: {e}, and the data may not have been fully saved")

    def save_pop(self, save_type='csv'):
        """
        保存当前代种群的结果
        :param save_type:  保存文件类型
        """
        # 获取算法与问题名称
        algo_name = type(self).__name__
        prob_name = type(self.problem).__name__
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        # 得到当前时间点时间戳
        timestamp = get_timestamp()
        if self.problem.num_obj == 1:  # 判断是否是单目标问题
            save_path = project_root + "\\Outputs\\Single\\" + algo_name + "_solve_" + prob_name + "_pop_" + timestamp
        else:
            save_path = project_root + "\\Outputs\\Multi\\" + algo_name + "_solve_" + prob_name + "_pop_" + timestamp
        # 若文件夹不存在则创建文件夹
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存问题和算法的相关信息
        info = dict()
        info['algorithm_params'] = self.get_params_info()
        info['problem_info'] = self.problem.get_info()
        save_json(info, save_path + "\\info")
        # 保存相关数据
        try:
            save_array(self.pop, save_path + "\\pop", save_type)
            save_array(self.objs, save_path + "\\objs", save_type)
            save_array(self.cons, save_path + "\\cons", save_type)
        except Exception as e:
            warnings.warn(f"There is a error with saving: {e}, and the data may not have been fully saved")

    def save_history(self, save_type='npz'):
        """
        保存种群所有历史的结果
        :param save_type:  保存文件类型
        """
        # 获取算法与问题名称
        algo_name = type(self).__name__
        prob_name = type(self.problem).__name__
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        # 得到当前时间点时间戳
        timestamp = get_timestamp()
        if self.problem.num_obj == 1:  # 判断是否是单目标问题
            save_path = (project_root + "\\Outputs\\Single\\" + algo_name + "_solve_" + prob_name +
                         "_history_" + timestamp)
        else:
            save_path = (project_root + "\\Outputs\\Multi\\" + algo_name + "_solve_" + prob_name +
                         "_history_" + timestamp)
        # 若文件夹不存在则创建文件夹
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存问题和算法的相关信息
        info = dict()
        info['algorithm_params'] = self.get_params_info()
        info['problem_info'] = self.problem.get_info()
        save_json(info, save_path + "\\info")
        # 保存相关数据
        try:
            # 预处理相关数据
            pop_dict = {f'iter_{i}': self.pop_history[i] for i in range(len(self.pop_history))}
            objs_dict = {f'iter_{i}': self.objs_history[i] for i in range(len(self.objs_history))}
            cons_dict = {f'iter_{i}': self.cons_history[i] for i in range(len(self.cons_history))}
            save_arrays(pop_dict, save_path + "\\pop_history", save_type)
            save_arrays(objs_dict, save_path + "\\objs_history", save_type)
            save_arrays(cons_dict, save_path + "\\cons_history", save_type)
        except Exception as e:
            warnings.warn(f"There is a error with saving: {e}, and the data may not have been fully saved")

    def load_best(self, file_path, save_type='csv'):
        """
        加载保存的最优个体解的结果
        :param file_path: 文件路径(Outputs文件夹下)
        :param save_type: 之前的保存结果 保存的类型
        """
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        file_path = project_root + "\\Outputs\\" + file_path  # 从输出文件夹中获取
        if not (self.best is None and self.best_obj is None and self.best_con is None):
            warnings.warn("The current best data will be overwritten")
        self.best = load_array(file_path + "\\best." + save_type)
        self.best_obj = load_array(file_path + "\\best_obj." + save_type)
        self.best_con = load_array(file_path + "\\best_con." + save_type)

    def load_pop(self, file_path, save_type='csv'):
        """
        加载保存的种群的结果
        :param file_path: 文件路径(Outputs文件夹下)
        :param save_type: 之前的保存结果 保存的类型
        """
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        file_path = project_root + "\\Outputs\\" + file_path  # 从输出文件夹中获取
        if not (self.pop is None and self.objs is None and self.cons is None):
            warnings.warn("The current pop data will be overwritten")
        self.pop = load_array(file_path + "\\pop." + save_type)
        self.objs = load_array(file_path + "\\objs." + save_type)
        self.cons = load_array(file_path + "\\cons." + save_type)

    def load_history(self, file_path, save_type='npz'):
        """
        加载保存的种群所有历史的结果
        :param file_path: 文件路径(Outputs文件夹下)
        :param save_type: 之前的保存结果 保存的类型
        """
        # 得到项目的根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]))
        file_path = project_root + "\\Outputs\\" + file_path  # 从输出文件夹中获取
        if len(self.pop_history) or len(self.objs_history) or len(self.cons_history):
            warnings.warn("The current history data will be overwritten")
        self.pop_history = list(load_arrays(file_path + "\\pop_history." + save_type).values())
        self.objs_history = list(load_arrays(file_path + "\\objs_history." + save_type).values())
        self.cons_history = list(load_arrays(file_path + "\\cons_history." + save_type).values())
