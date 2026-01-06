from Problems.Multi import ZDT1
from Problems.Multi import DTLZ2
from Algorithms.Multi import MOEAD
"""MOEA/D算法调用测试"""

if __name__ == '__main__':
    problem = DTLZ2()
    algorithm = MOEAD(pop_size=100, max_iter=100, show_mode='obj')
    algorithm.solve(problem)
    print("HV: ", algorithm.cal_score('HV'))
    print("GD: ", algorithm.cal_score('GD'))
    print("IGD: ", algorithm.cal_score('IGD'))
    print("time(s): ", algorithm.run_time)
    algorithm.plot(show_mode='obj')
    algorithm.plot_scores()

