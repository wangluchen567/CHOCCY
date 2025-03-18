from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Mutations import *


class SA(ALGORITHM):
    def __init__(self, num_pop=1, num_iter=10000, init_temp=1e4, alpha=0.99, perturb_prob=0.5, show_mode=0):
        """
        模拟退火算法
        *Code Author: Luchen Wang
        :param num_iter: 迭代次数
        :param init_temp: 初始温度
        :param alpha: 温度衰减系数
        :param perturb_prob: 扰动概率(变异概率)
        :param show_mode: 绘图模式
        """
        super().__init__(num_pop, num_iter, None, perturb_prob, None, show_mode)
        self.only_solve_single = True
        self.init_temp = init_temp
        self.temp = self.init_temp
        self.alpha = alpha
        self.p = None
        self.obj = None
        self.con = None
        self.fit = None

    def init_algorithm(self, problem, pop=None):
        super().init_algorithm(problem, pop)
        self.p = self.pop[0].reshape(1, -1)
        self.obj = self.cal_objs(self.p)
        self.con = self.cal_cons(self.p)
        self.fit = self.cal_fits(self.obj, self.con)

    @ALGORITHM.record_time
    def run_step(self, i):
        if self.temp < 1e-100:
            # 若温度已经很小，则不再更新
            pass
        else:
            for i in range(len(self.pop)):
                self.run_one()
                self.pop[i], self.objs[i], self.cons[i], self.fits[i] \
                    = self.p, self.obj, self.con, self.fit
        # 记录每步状态
        self.record()

    def run_one(self):
        """对单个个体进行扰动和优化"""
        # 对个体解进行扰动
        new_p = self.perturb(self.p)
        # 得到扰动解的目标值与约束值
        # 得到扰动解的目标值与约束值
        new_obj = self.cal_objs(new_p)
        new_con = self.cal_cons(new_p)
        new_fit = self.cal_fits(new_obj, new_con)
        # 使用metrospolis接受准则接受解
        if self.metrospolis(self.fit[0], new_fit[0], self.temp):
            # 更新解集
            self.p = new_p
            self.obj = new_obj
            self.con = new_con
            self.fit = new_fit
        # 更新温度
        self.temp = self.alpha * self.temp

    @staticmethod
    def metrospolis(old, new, temp):
        """
        使用metrospolis接受准则接受解
        :param old: 扰动前旧的解(适应度值)
        :param new: 扰动得到的新解(适应度值)
        :param temp: 当前温度
        :return: 是否接受新解
        """
        # 计算能量差
        delta_e = new - old
        if delta_e < 0:
            # 若新解比旧解更好则直接接受新解
            return True
        elif temp < 1e-100:
            # 温度太低之后直接不接受新解
            return False
        else:
            # 若新解比旧解更差则以一定概率接受新解
            return np.random.rand() < np.exp(-delta_e / temp)

    @staticmethod
    def metrospolis_multi(old, new, temp):
        """
        使用metrospolis接受准则接受解(多个解)
        :param old: 扰动前旧的解(适应度值)
        :param new: 扰动得到的新解(适应度值)
        :param temp: 当前温度
        :return: 是否接受新解
        """
        accept_prob = np.ones(len(old))
        # 若新解比旧解更好则直接接受新解
        # 若新解比旧解更差则以一定概率接受新解
        worse = np.array(old < new)
        accept_prob[worse] = np.exp((old[worse] - new[worse]) / temp)
        accept = np.random.uniform(0, 1, accept_prob.shape) < accept_prob
        return accept

    def perturb(self, solutions):
        """对解进行扰动"""
        # 防止影响原数据
        new_solutions = solutions.copy()
        for t in self.unique_type:
            new_solutions[:, self.type_indices[t]] = self.mutate_(t, new_solutions[:, self.type_indices[t]],
                                                                  self.lower[self.type_indices[t]],
                                                                  self.upper[self.type_indices[t]],
                                                                  self.mutate_prob)
        return new_solutions

    @staticmethod
    def mutate_(problem_type, solutions, lower, upper, mutate_prob):
        """根据变量的不同类型对解进行扰动(变异)"""
        if problem_type == ALGORITHM.REAL:
            return polynomial_mutation(solutions, lower, upper, mutate_prob)
        elif problem_type == ALGORITHM.INT:
            return polynomial_mutation(solutions, lower, upper, mutate_prob)
        elif problem_type == ALGORITHM.BIN:
            return bit_mutation(solutions, mutate_prob)
        elif problem_type == ALGORITHM.PMU:
            return exchange_mutation(solutions, mutate_prob)
        elif problem_type == ALGORITHM.FIX:
            return fix_label_mutation(solutions, mutate_prob)
        else:
            raise ValueError("The problem type does not exist")
