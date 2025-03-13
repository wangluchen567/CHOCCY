import numpy as np
from tqdm import tqdm
from typing import Union
from Problems.PROBLEM import PROBLEM
from Metrics.Hypervolume import cal_hv
from Algorithms.Utility.Utils import fast_nd_sort, shuffle_matrix_in_row, record_time
from Algorithms.Utility.Plots import plot_scores, plot_data, plot_objs, plot_decs_objs
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
    NULL = -1  # 不绘制
    BAR = 0  # 绘制进度条
    OBJ = 1  # 绘制目标空间
    DEC = 2  # 绘制决策空间
    OAD2 = 3  # 绘制目标空间和决策空间混合(等高线)
    OAD3 = 4  # 绘制目标空间和决策空间混合(三维空间)
    PRB = 5  # 问题提供绘图方法
    ALG = 6  # 算法提供绘图方法

    def __init__(self,
                 problem: PROBLEM,
                 num_pop: int = 100,
                 num_iter: int = 100,
                 cross_prob: Union[float, None] = None,
                 mutate_prob: Union[float, None] = None,
                 educate_prob: Union[float, None] = None,
                 show_mode: int = BAR):
        """
        算法父类
        *Code Author: LuChen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param educate_prob: 教育概率
        :param show_mode: 绘图模式
        """
        self.problem = problem
        self.num_dec = self.problem.num_dec
        self.num_obj = self.problem.num_obj
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.problem_type = self.problem.problem_type
        self.unique_type = self.problem.unique_type
        self.type_indices = self.problem.type_indices
        self.lower = self.problem.lower
        self.upper = self.problem.upper
        self.show_mode = show_mode
        # 初始化交叉、变异和教育概率
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.educate_prob = educate_prob
        # 算法可求解的问题类型

        # 初始化种群解及其目标/约束/适应度
        self.pop, self.objs, self.cons, self.fits = None, None, None, None
        # 记录种群解及其目标/约束(适应度为中间值不记录)
        self.pop_history, self.objs_history, self.cons_history = [], [], []
        # 初始化种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best, self.best_obj, self.best_con = None, np.inf, np.inf
        # 记录种群最优个体及其目标/约束(适应度为中间值不记录)
        self.best_history, self.best_obj_his, self.best_con_his = [], [], []
        # 初始化迭代器
        self.iterator = None
        # 记录评价指标
        self.scores = np.empty(0)
        # 记录运行时间
        self.run_time = 0.0
        self.record_t = []
        # 决策变量示例(固定标签问题)
        self.example_dec = None
        if hasattr(problem, 'example_dec'):
            self.example_dec = problem.example_dec

    @record_time
    def init_algorithm(self):
        """初始化算法"""
        # 初始化种群与参数
        self.init_params()
        # 初始化种群并对种群进行评价
        self.init_and_eval()
        # 构建迭代器
        self.iterator = tqdm(range(self.num_iter)) if self.show_mode == 0 else range(self.num_iter)

    @record_time
    def init_algorithm_with(self, pop=None):
        """通过给定种群进行初始化"""
        # 初始化参数(各个概率参数)
        self.init_params()
        # 初始化种群并对种群进行评价
        self.init_and_eval(pop)
        # 构建迭代器
        self.iterator = tqdm(range(self.num_iter)) if self.show_mode == 0 else range(self.num_iter)

    def init_params(self):
        """初始化参数(交叉、变异和教育概率)"""
        self.cross_prob = 1.0 if self.cross_prob is None else self.cross_prob
        self.mutate_prob = 1 / self.num_dec if self.mutate_prob is None else self.mutate_prob
        self.educate_prob = 0.5 if self.educate_prob is None else self.educate_prob

    def init_and_eval(self, pop=None):
        """初始化种群并对种群中解进行评价"""
        self.pop = self.init_pop() if pop is None else pop
        # 对种群中解进行评价(求目标值/约束值/适应度)
        self.objs, self.cons, self.fits = self.evaluate(self.pop)
        # 记录当前种群信息
        self.record()

    def cal_objs(self, decs):
        """计算目标值"""
        return self.problem.cal_objs(decs)

    def cal_cons(self, decs):
        """计算约束值"""
        return self.problem.cal_cons(decs)

    @staticmethod
    def cal_objs_based_cons(objs, cons):
        """计算约束松弛后的目标值"""
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
        """根据给定目标值和约束值得到适应度值(默认是单目标情况)"""
        # 检查是否均满足约束，若均满足约束则无需考虑约束
        if np.all(cons <= 0):
            return objs.flatten()
        else:
            return self.cal_objs_based_cons(objs, cons).flatten()

    def evaluate(self, pop):
        """给定种群解并对解进行评价(求目标值/约束值/适应度)"""
        objs = self.cal_objs(pop)
        cons = self.cal_cons(pop)
        fits = self.cal_fits(objs, cons)
        return objs, cons, fits

    @staticmethod
    def record_time(method):
        """统计运行时间"""
        return record_time(method)

    def run(self):
        """运行算法(主函数)"""
        raise NotImplemented

    def run_step(self, *args, **kwargs):
        """运行算法单步"""
        self.record()

    def init_pop(self):
        """初始化种群"""
        init_dict = {ALGORITHM.REAL: self.init_pop_real,
                     ALGORITHM.INT: self.init_pop_integer,
                     ALGORITHM.BIN: self.init_pop_binary,
                     ALGORITHM.PMU: self.init_pop_permutation,
                     ALGORITHM.FIX: self.init_pop_fix_label}
        pop = np.zeros((self.num_pop, self.num_dec))
        # 若没有实数或整数部分则直接初始化为整型
        if np.all(self.unique_type > 1):
            pop = np.zeros((self.num_pop, self.num_dec), dtype=int)
        # 遍历所有问题类型
        for t in self.unique_type:
            pop[:, self.type_indices[t]] = init_dict.get(t)()
        return pop

    def init_pop_real(self):
        """初始化求解实数或整数问题的种群"""
        pop = np.random.uniform(self.lower[self.type_indices[ALGORITHM.REAL]],
                                self.upper[self.type_indices[ALGORITHM.REAL]],
                                size=(self.num_pop, len(self.type_indices[ALGORITHM.REAL])))
        return pop

    def init_pop_integer(self):
        """初始化求解实数或整数问题的种群"""
        pop = np.random.uniform(self.lower[self.type_indices[ALGORITHM.INT]],
                                self.upper[self.type_indices[ALGORITHM.INT]],
                                size=(self.num_pop, len(self.type_indices[ALGORITHM.INT])))
        return pop

    def init_pop_binary(self):
        """初始化求解二进制问题的种群"""
        pop = np.random.randint(2, size=(self.num_pop, len(self.type_indices[ALGORITHM.BIN])))
        return pop

    def init_pop_permutation(self):
        """初始化求解序列问题的种群"""
        pop = np.argsort(np.random.uniform(0, 1,
                                           size=(self.num_pop, len(self.type_indices[ALGORITHM.PMU]))), axis=1)
        return pop

    def init_pop_fix_label(self):
        """初始化求解固定标签问题的种群"""
        # 确保给定的示例和决策向量大小相等
        if len(self.type_indices[ALGORITHM.FIX]) != len(self.example_dec):
            raise ValueError("The given example and decision vector are not of equal size")
        # 确定初始向量
        pop = self.example_dec.copy()
        # 初始化种群向量
        pop = np.repeat(pop.reshape(1, -1), self.num_pop, axis=0)
        # 打乱每行的个体向量
        shuffle_matrix_in_row(pop)
        # 确保为整型
        pop = np.array(pop, dtype=int)
        return pop

    def operator(self, mating_pool):
        """进行交叉变异生成子代"""
        offspring = self.pop[mating_pool]
        for t in self.unique_type:
            offspring[:, self.type_indices[t]] = self.operator_(t)(offspring[:, self.type_indices[t]],
                                                                   self.lower[self.type_indices[t]],
                                                                   self.upper[self.type_indices[t]],
                                                                   self.cross_prob, self.mutate_prob)
        return offspring

    @staticmethod
    def operator_(problem_type):
        """根据问题类型返回对应函数"""
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
        """交配池选择"""
        if num_next is None:
            num_next = self.num_pop
        if k >= 2:
            # 使用锦标赛选择获取交配池
            return tournament_selection(self.fits, num_next, k)
        else:
            # 使用轮盘赌选择获取交配池
            return roulette_selection(self.fits, num_next)

    def pop_merge(self, offspring):
        """当前种群与其子代合并"""
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
        """进行环境选择"""
        # 将当前种群与其子代合并
        new_pop, new_objs, new_cons, new_fits = self.pop_merge(offspring)
        # 使用选择策略(默认精英选择)选择进入下一代新种群的个体
        best_indices = elitist_selection(new_fits, self.num_pop)
        self.pop = new_pop[best_indices]
        self.objs = new_objs[best_indices]
        self.cons = new_cons[best_indices]
        self.fits = new_fits[best_indices]

    def get_best(self):
        """获取当前种群的最优解"""
        self.best, self.best_obj, self.best_con = self.get_best_(self.pop, self.objs, self.cons)

    @staticmethod
    def get_best_(pop, objs, cons):
        """获取给定种群的最优解"""
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
        self.get_best()
        self.best_history.append(self.best)
        self.best_obj_his.append(self.best_obj)
        self.best_con_his.append(self.best_con)
        if self.num_obj == 1:
            # 若是单目标问题则直接记录分数(低复杂度)
            self.record_score()

    def record_score(self):
        """记录分数值"""
        if self.num_obj == 1:
            self.scores = self.best_obj_his
        else:
            hv_value = cal_hv(self.best_obj, self.problem.optimums)
            self.scores = np.append(self.scores, hv_value)

    def plot(self, show_mode=None, n_iter=None, pause=False):
        """
        绘图函数，根据不同模式进行绘图
        (-1:不绘制, 0:进度条, 1:目标空间, 2:决策空间, 3:混合模式(等高线),
        4:混合模式(三维空间), 5:问题提供, 6:算法提供)
        """
        if show_mode is not None:
            self.show_mode = show_mode
        if self.show_mode == self.NULL or self.show_mode == self.BAR:
            pass
        elif self.show_mode == self.OBJ:
            self.plot_objs(n_iter, pause)
        elif self.show_mode == self.DEC:
            self.plot_pop(n_iter, pause)
        elif self.show_mode == self.OAD2:
            self.plot_decs_objs(n_iter, pause)
        elif self.show_mode == self.OAD3:
            self.plot_decs_objs(n_iter, pause, contour=False)
        elif self.show_mode == self.PRB:
            self.plot_by_problem(n_iter, pause)
        elif self.show_mode == self.ALG:
            self.plot_(n_iter, pause)
        else:
            raise ValueError("There is no such plotting mode")

    def plot_(self, n_iter, pause):
        """提供算法自定义绘图的接口"""
        pass

    def plot_pop(self, n_iter=None, pause=False, pause_time=0.1):
        """绘制种群个体决策向量"""
        if pause or n_iter is None:
            plot_data(self.pop, n_iter, pause, pause_time)
        else:
            plot_data(self.pop_history[n_iter], n_iter, pause, pause_time)

    def plot_objs(self, n_iter=None, pause=False, pause_time=0.1):
        """绘制种群目标值"""
        if self.num_obj == 1:
            # 若是单目标问题，绘制目标值范围情况
            plot_objs(self.objs_history, n_iter, pause, pause_time)
        else:
            plot_objs(self.objs, n_iter, pause, pause_time, self.problem.pareto_front)

    def plot_decs_objs(self, n_iter=None, pause=False, pause_time=0.1, contour=True, sym=True):
        """在特定条件下可同时绘制决策向量与目标值"""
        if pause or n_iter is None:
            plot_decs_objs(self.problem, self.pop, self.objs,
                           n_iter, pause, pause_time, contour=contour, sym=sym)
        else:
            plot_decs_objs(self.problem, self.pop_history[n_iter], self.objs_history[n_iter],
                           n_iter, pause, pause_time, contour=contour, sym=sym)

    def plot_by_problem(self, n_iter=None, pause=False):
        """使用问题给定的绘图函数绘图"""
        if pause or n_iter is None:
            self.problem.plot(self.best_history[-1], n_iter, pause)
        else:
            self.problem.plot(self.best_history[n_iter], n_iter, pause)

    def get_scores(self):
        """获取历史所有种群的评价分数"""
        # 若是单目标问题则评价分数就是最优目标值
        if self.num_obj == 1:
            self.scores = self.best_obj_his
            return self.scores
        # 若是多目标问题则计算评价分数
        self.scores = np.zeros(len(self.best_obj_his))
        for i in range(len(self.best_obj_his)):
            self.scores[i] = cal_hv(self.best_obj_his[i], self.problem.optimums)
        return self.scores

    def plot_scores(self, score_type=None):
        """绘制指标的变化情况"""
        if score_type is None:
            if self.num_obj == 1:
                score_type = "Fitness"
            else:
                score_type = "HV"
        if self.scores is None or len(self.scores) == 0:
            self.get_scores()
        plot_scores(self.scores, score_type)
