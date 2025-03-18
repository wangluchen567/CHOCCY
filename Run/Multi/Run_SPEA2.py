from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.SPEA2 import SPEA2
import cProfile

if __name__ == '__main__':
    problem = ZDT1()
    alg = SPEA2(num_pop=100, num_iter=100, show_mode=1)
    cProfile.run("alg.solve(problem)", sort='cumulative')
    print("HV: ", alg.cal_score('HV'))
    print("GD: ", alg.cal_score('GD'))
    print("IGD: ", alg.cal_score('IGD'))
    print("time(s): ", alg.run_time)
    alg.plot(show_mode=1)
    alg.plot_scores()
