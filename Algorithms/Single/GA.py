from Algorithms.ALGORITHM import ALGORITHM


class GA(ALGORITHM):
    def __init__(self, pop_size=100, max_iter=100, cross_prob=None, mutate_prob=None, show_mode=0):
        """
        遗传算法
        *Code Author: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, None, show_mode)
        self.only_solve_single = True

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取交配池
        mating_pool = self.mating_pool_selection()
        # 交叉变异生成子代
        offspring = self.operator(mating_pool)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record()
