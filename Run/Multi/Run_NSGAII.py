from Problems.Multi import ZDT3
from Problems.Multi import DTLZ2
from Problems.Multi import MOKP
from Algorithms.Multi import NSGAII

if __name__ == '__main__':
    problem = ZDT3()
    algorithm = NSGAII(pop_size=100, max_iter=100, show_mode=0)
    algorithm.set_score_type('IGD')
    algorithm.solve(problem)
    print("HV: ", algorithm.cal_score('HV'))
    print("GD: ", algorithm.cal_score('GD'))
    print("IGD: ", algorithm.cal_score('IGD'))
    print("GD+: ", algorithm.cal_score('GD+'))
    print("IGD+: ", algorithm.cal_score('IGD+'))
    print("time(s): ", algorithm.run_time)
    algorithm.plot(show_mode=1)
    algorithm.plot_scores()
