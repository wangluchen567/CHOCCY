import numpy as np
from Problems import PROBLEM
from Algorithms import View
from Algorithms.Single import DE


class QuadProblem(PROBLEM):

    def __init__(self, num_dec=30, upper=100):
        """
        定义一个带约束的二次规划问题
        :param num_dec: 决策变量个数
        :param upper: 决策变量上界
        """
        num_obj = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, 0, upper)

    def _cal_objs(self, X):
        objs = np.sum(X ** 2, axis=1)
        return objs

    def _cal_cons(self, X):
        return 1 - np.sum(X, axis=1)


if __name__ == '__main__':
    problem = QuadProblem(num_dec=10)  # 实例化问题，并指定决策向量大小
    # 实例化算法并设置种群大小为100，迭代次数为100，优化过程展示为目标值变化情况
    algorithm = DE(pop_size=100, max_iter=100, show_mode=View.LOG)
    algorithm.solve(problem)  # 使用该算法求解问题
    # 获取最优解并打印
    best, best_obj, best_con = algorithm.get_best()
    print("最优解：", best)
    print("最优解的目标值：", best_obj)
    print("最优解的约束值：", best_con)
    print("算法运行时间(秒)：", algorithm.run_time)