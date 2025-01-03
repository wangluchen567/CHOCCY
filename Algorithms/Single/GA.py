from Algorithms.ALGORITHM import ALGORITHM


class GA(ALGORITHM):
    def __init__(self, problem, num_pop, num_iter, cross_prob=None, mutate_prob=None, show_mode=None):
        """
        遗传算法
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
            # 获取交配池
            mating_pool = self.mating_pool_selection()
            # 交叉变异生成子代
            offspring = self.operator(mating_pool)
            # 进行环境选择
            self.environmental_selection(offspring)
            # 记录每步状态
            self.record(i + 1)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)
