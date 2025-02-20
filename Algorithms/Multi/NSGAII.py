import numpy as np
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Utils import fast_nd_sort, cal_crowd_dist, cal_fitness


class NSGAII(ALGORITHM):
    def __init__(self, problem, num_pop=100, num_iter=100, cross_prob=None, mutate_prob=None, show_mode=None):
        """
        This code is based on the research presented in
        "A fast and elitist multi-objective genetic algorithm: NSGA-II"
        by K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, show_mode)
        self.init_algorithm()

    @ALGORITHM.record_time
    def run(self):
        """运行算法(主函数)"""
        # 绘制初始状态图
        self.plot(pause=True, n_iter=0)
        for i in self.iterator:
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)

    def run_step(self, i):
        """运行算法单步"""
        # 获取交配池
        mating_pool = self.mating_pool_selection()
        # 交叉变异生成子代
        offspring = self.operator(mating_pool)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record(i + 1)

    def get_fitness(self, objs, cons):
        """根据给定目标值和约束值得到适应度值"""
        # 检查是否均满足约束，若均满足约束则无需考虑约束
        if np.all(cons <= 0):
            objs_based_cons = objs
        else:
            objs_based_cons = self.cal_objs_based_cons(objs, cons)
        # 对于多目标问题则需要考虑所在前沿面及拥挤度情况
        fronts, ranks = fast_nd_sort(objs_based_cons)
        crowd_dist = cal_crowd_dist(objs, fronts)
        fitness = cal_fitness(ranks, crowd_dist)

        return fitness
