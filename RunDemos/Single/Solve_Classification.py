import numpy as np
from Algorithms import View
from Algorithms.Single import GA, GD, Adam
from Problems.Single import Classification


def GA_solve_(problem, max_iter=100):
    algorithm = GA(max_iter=max_iter, show_mode=View.PROB)
    algorithm.solve(problem)
    # 获取最优解并打印
    best, best_obj, best_con = algorithm.get_best()
    print("最优解：", best)
    print("最优解的目标值：", best_obj)
    print("算法运行时间(秒)：", algorithm.run_time)


def GD_solve_(problem, max_iter=100):
    algorithm = GD(max_iter=max_iter, learning_rate=0.9, show_mode=View.PROB)
    algorithm.solve(problem)
    # 获取最优解并打印
    best, best_obj, best_con = algorithm.get_best()
    print("最优解：", best)
    print("最优解的目标值：", best_obj)
    print("算法运行时间(秒)：", algorithm.run_time)


def Adam_solve_(problem, max_iter=100):
    algorithm = Adam(max_iter=max_iter, learning_rate=0.5, show_mode=View.PROB)
    algorithm.solve(problem)
    # 获取最优解并打印
    best, best_obj, best_con = algorithm.get_best()
    print("最优解：", best)
    print("最优解的目标值：", best_obj)
    print("算法运行时间(秒)：", algorithm.run_time)


if __name__ == '__main__':
    np.random.seed(0)
    problem = Classification(num_dec=3)
    # GA_solve_(problem)
    # GD_solve_(problem)
    Adam_solve_(problem)
