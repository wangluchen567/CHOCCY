from Problems.Multi import ZDT1
from Problems.Multi import DTLZ2
from Problems.Multi import MOKP
from Algorithms.Multi import SPEA2
import cProfile

if __name__ == '__main__':
    problem = ZDT1()
    algorithm = SPEA2(pop_size=100, max_iter=100, show_mode=1)
    cProfile.run("algorithm.solve(problem)", sort='cumulative')
    print("HV: ", algorithm.cal_score('HV'))
    print("GD: ", algorithm.cal_score('GD'))
    print("IGD: ", algorithm.cal_score('IGD'))
    print("GD+: ", algorithm.cal_score('GD+'))
    print("IGD+: ", algorithm.cal_score('IGD+'))
    print("time(s): ", algorithm.run_time)
    algorithm.plot(show_mode=1)
    algorithm.plot_scores()
