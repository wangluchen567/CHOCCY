from Algorithms.Multi import NNDREA
from Problems.Multi import MOKP
"""NNDREA算法调用测试"""

if __name__ == '__main__':
    problem = MOKP(10000)
    algorithm = NNDREA(pop_size=100, max_iter=100, show_mode='obj')
    algorithm.solve(problem)
    print("HV: ", algorithm.cal_score('HV'))
    print("time(s): ", algorithm.run_time)
    print(algorithm.get_params_info())
    algorithm.plot(show_mode='obj')
    algorithm.plot_scores()
