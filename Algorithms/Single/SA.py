from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Mutations import *


class SA(ALGORITHM):
    def __init__(self, problem, num_pop, num_iter, init_temp=1000, alpha=0.95, perturb_prob=0.5, show_mode=None):
        """
        模拟退火算法
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param init_temp: 初始温度
        :param alpha: 温度衰减系数
        :param perturb_prob: 扰动概率(变异概率)
        :param show_mode: 绘图模式
        """
        super().__init__(problem, num_pop, num_iter, None, perturb_prob, show_mode)
        self.init_temp = init_temp
        self.alpha = alpha
        self.init_algorithm()

    def run(self):
        # 获取初始化温度
        temp = self.init_temp
        # 绘制初始状态图
        self.plot(pause=True, n_iter=0)
        for i in self.iterator:
            # 对解进行扰动
            new_pop = self.perturb(self.pop)
            # 得到扰动解的目标值与约束值
            new_objs = self.cal_objs(new_pop)
            new_cons = self.cal_cons(new_pop)
            new_fits = self.get_fitness(new_objs, new_cons)
            # 使用Metrospolis接受准则接受解
            accept = self.Metrospolis(self.fitness, new_fits, temp)
            # 更新解集
            self.pop[accept] = new_pop[accept]
            self.objs[accept] = new_objs[accept]
            self.cons[accept] = new_cons[accept]
            self.fitness[accept] = new_fits[accept]
            # 更新温度
            temp = self.alpha * temp
            # 检查温度是否已经很小
            if temp == 0:
                print("The temperature is already very low, algorithm stopped early")
                break
            # 记录每步状态
            self.record(i + 1)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)

    @staticmethod
    def Metrospolis(old, new, temp):
        """
        使用Metrospolis接受准则接受解
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
        if problem_type == 0:
            return polynomial_mutation(solutions, lower, upper, mutate_prob)
        elif problem_type == 1:
            return bit_mutation(solutions, mutate_prob)
        elif problem_type == 2:
            return exchange_mutation(solutions, mutate_prob)
        elif problem_type == 3:
            return fix_label_mutation(solutions, mutate_prob)
        else:
            raise ValueError("The problem type does not exist")
