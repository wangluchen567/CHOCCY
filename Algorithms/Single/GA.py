import numpy as np
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Selections import tournament_selection



class GA(ALGORITHM):
    def __init__(self, problem, num_pop, num_iter, cross_prob=None, mutate_prob=None, show_mode=None):
        """
        遗传算法
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式 (0:不绘制图像, 1:目标空间, 2:决策空间, 3:混合模式)
        """
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, show_mode)
        self.init_algorithm()

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
            # 记录每步状态
            self.record()
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)

    def selection(self, k=2):
        # 使用锦标赛选择获取交配池
        return tournament_selection(self.objs, k)

    def environmental_selection(self, offspring):
        # 进行环境选择
        # 先计算子代目标值与
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 将父代与子代合并获得新种群
        new_pop = np.vstack((self.pop, offspring))
        new_objs = np.vstack((self.objs, off_objs))
        new_cons = np.vstack((self.cons, off_cons))
        # 根据目标值对种群中的个体进行排序
        index_sort = np.argsort(new_objs.flatten())
        # 取目标值最优的个体组成新的种群
        self.pop = new_pop[index_sort][:self.num_pop]
        self.objs = new_objs[index_sort][:self.num_pop]
        self.cons = new_cons[index_sort][:self.num_pop]
