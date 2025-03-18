from Problems.Multi.ZDT.ZDT3 import ZDT3
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NSGAII import NSGAII

if __name__ == '__main__':
    problem = ZDT3()
    alg = NSGAII(problem, num_pop=100, num_iter=100, show_mode=1)
    alg.set_score_type('IGD')
    alg.run()
    print("HV: ", alg.cal_score('HV'))
    print("GD: ", alg.cal_score('GD'))
    print("IGD: ", alg.cal_score('IGD'))
    print("time(s): ", alg.run_time)
    alg.plot(show_mode=1)
    alg.plot_scores()
