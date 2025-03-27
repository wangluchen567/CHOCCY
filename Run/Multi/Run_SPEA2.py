from Problems.Multi import ZDT1
from Problems.Multi import DTLZ2
from Problems.Multi import MOKP
from Algorithms.Multi import SPEA2
import cProfile

if __name__ == '__main__':
    problem = ZDT1()
    alg = SPEA2(pop_size=100, max_iter=100, show_mode=1)
    cProfile.run("alg.solve(problem)", sort='cumulative')
    print("HV: ", alg.cal_score('HV'))
    print("GD: ", alg.cal_score('GD'))
    print("IGD: ", alg.cal_score('IGD'))
    print("time(s): ", alg.run_time)
    alg.plot(show_mode=1)
    alg.plot_scores()
