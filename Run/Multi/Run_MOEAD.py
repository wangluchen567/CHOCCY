from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Algorithms.Multi.MOEAD import MOEAD

if __name__ == '__main__':
    problem = DTLZ2()
    alg = MOEAD(problem, num_pop=100, num_iter=100, func_type=1, show_mode=1)
    alg.run()
    print("HV: ", alg.cal_score('HV'))
    print("GD: ", alg.cal_score('GD'))
    print("IGD: ", alg.cal_score('IGD'))
    print("time(s): ", alg.run_time)
    alg.plot(show_mode=1)
    alg.plot_scores()

