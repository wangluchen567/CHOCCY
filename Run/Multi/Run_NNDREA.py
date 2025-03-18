from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NNDREA import NNDREA

if __name__ == '__main__':
    problem = MOKP(10000)
    alg = NNDREA(problem, num_pop=100, num_iter=200, show_mode=1)
    alg.run()
    print("HV: ", alg.cal_score('HV'))
    print("time(s): ", alg.run_time)
    alg.plot(show_mode=1)
    alg.plot_scores()
