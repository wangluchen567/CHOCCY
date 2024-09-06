import numpy as np
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Selections import tournament_selection
from Algorithms.Utility.Utils import fast_nd_sort, cal_crowd_dist, cal_fitness
from Algorithms.Utility.Operators import operator_real, operator_binary, operator_permutation, operator_fix_label


class NSGAII(ALGORITHM):
    def __init__(self, problem, num_pop, num_iter, cross_prob=None, mutate_prob=None, show_mode=None):
        """
        A fast and elitist multi-objective genetic algorithm: NSGA-II
        (K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan)
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式 (0:不绘制图像, 1:目标空间, 2:决策空间)
        """
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, show_mode)
        self.init_algorithm()
        # 计算每个个体的等价适应度值
        self.fitness = self.get_fitness(self.objs)

    @ALGORITHM.record_time
    def run(self):
        """运行算法(主函数)"""
        # 绘制初始状态图
        self.plot(pause=True, n_iter=0)
        for i in self.iterator:
            # 获取交配池
            mating_pool = self.selection()
            # 交叉变异生成子代
            offspring = self.operator(mating_pool)
            # 进行环境选择
            self.environmental_selection(offspring)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)
            # 记录每步状态
            self.record()

    def get_fitness(self, objs):
        """计算每个个体的等价适应度值"""
        fronts, ranks = fast_nd_sort(objs)
        crowd_dist = cal_crowd_dist(objs, fronts)
        fitness = cal_fitness(ranks, crowd_dist)
        return fitness

    def selection(self, k=2):
        return tournament_selection(self.fitness, k)

    def environmental_selection(self, offspring):
        # 进行环境选择(多目标)
        # 先计算子代目标值
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 将父代与子代合并获得新种群
        new_pop = np.vstack((self.pop, offspring))
        new_objs = np.vstack((self.objs, off_objs))
        new_cons = np.vstack((self.cons, off_cons))
        # 重新计算合并种群的的等价适应度值
        fitness = self.get_fitness(new_objs)
        # 根据适应度值对种群中的个体进行排序
        index_sort = np.argsort(fitness)
        # 取适应度值最优的个体组成新的种群
        self.pop = new_pop[index_sort][:self.num_pop]
        self.objs = new_objs[index_sort][:self.num_pop]
        self.cons = new_cons[index_sort][:self.num_pop]
        self.fitness = fitness[index_sort][:self.num_pop]
