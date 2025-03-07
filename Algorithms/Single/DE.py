from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Operators import operator_de_1


class DE(ALGORITHM):
    def __init__(self, problem, num_pop=100, num_iter=100, cross_prob=None, mutate_prob=None, factor=0.5, show_mode=0):
        """
        差分进化算法
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, None, show_mode)
        self.factor = factor

    @ALGORITHM.record_time
    def run(self):
        """运行算法(主函数)"""
        # 初始化算法
        self.init_algorithm()
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
        mating_pool = self.mating_pool_selection(2 * self.num_pop)
        # 交叉变异生成子代
        offspring = self.operator(mating_pool)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record(i + 1)

    def operator(self, mating_pool):
        """重写算子为差分进化算子"""
        return operator_de_1(self.pop, self.pop[mating_pool], self.lower, self.upper,
                             self.cross_prob, self.mutate_prob, self.factor)
