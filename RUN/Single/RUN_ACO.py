from Problems.Single.TSP import TSP
from Algorithms.Single.ACO import ACO

if __name__ == '__main__':
    problem = TSP(30)
    alg = ACO(problem, num_pop=20, num_iter=100, show_mode=1)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()
