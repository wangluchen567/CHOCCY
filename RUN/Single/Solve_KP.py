import numpy as np
import seaborn as sns
from Problems.Single.KP import KP
from Algorithms.Single.GA import GA
from Algorithms.Single.DP_KP import DP_KP
from Algorithms.Single.Greedy_KP import Greedy_KP
from Algorithms.Single.NNDREAS import NNDREAS


def DP_Solve_KP(problem):
    alg = DP_KP(problem)
    alg.run()
    print("DP Solve KP")
    print("Run time: ", alg.run_time)
    print("best value: ", alg.best_obj[0])


def Greedy_Solve_KP(problem):
    alg = Greedy_KP(problem)
    alg.run()
    print("Greedy Solve KP")
    print("Run time: ", alg.run_time)
    print("best value: ", alg.best_obj[0])


def GA_Solve_KP(problem, num_pop=100, num_iter=100):
    alg = GA(problem, num_pop=num_pop, num_iter=num_iter, show_mode=0)
    alg.run()
    print("GA Solve KP")
    print("Run time: ", alg.run_time)
    print("best value: ", alg.best_obj[0])
    # alg.plot_scores()


def NNDREA_Solve_KP(problem, num_pop=100, num_iter=100):
    alg = NNDREAS(problem, num_pop=num_pop, num_iter=num_iter, show_mode=0)
    alg.run()
    print("NNDREA Solve KP")
    print("Run time: ", alg.run_time)
    print("best value: ", alg.best_obj[0])
    print(np.min(alg.pop_weights), np.max(alg.pop_weights))

    sns.heatmap(alg.pop_weights, cmap="YlGnBu")
    alg.plot_scores()


if __name__ == '__main__':
    problem = KP(num_dec=1000)
    # DP_Solve_KP(problem)
    Greedy_Solve_KP(problem)
    GA_Solve_KP(problem)
    NNDREA_Solve_KP(problem)
