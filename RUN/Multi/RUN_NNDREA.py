from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NNDREA import NNDREA

if __name__ == '__main__':
    problem = MOKP(10000)
    alg = NNDREA(problem, num_pop=100, num_iter=1000, show_mode=0)
    alg.run()
    print("Run time: ", alg.run_time)
    alg.plot(show_mode=1)
    print("Final score: ", alg.get_scores()[-1])
    alg.plot_scores()