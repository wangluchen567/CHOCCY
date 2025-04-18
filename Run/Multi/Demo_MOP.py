from Problems.Multi import MOP2
from Algorithms.Multi import MOEAD
from Algorithms.Multi import NSGAII
from Algorithms.Multi import SPEA2


if __name__ == '__main__':
    problem = MOP2()
    algorithm = NSGAII(pop_size=100, max_iter=100, show_mode=1)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print("最优解集：", best)
    print("最优解集的目标值：", best_obj)
    print("算法运行时间(秒)：", algorithm.run_time)