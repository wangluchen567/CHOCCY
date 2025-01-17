from Problems.Single.TSP import TSP
from Algorithms.Single.GA import GA
from Algorithms.Single.ACO import ACO
from Algorithms.Single.FI import FI
from Algorithms.Single.GFLS import GFLS

if __name__ == '__main__':
    problem = TSP(100)
    alg = GA(problem, num_pop=100, num_iter=1000, show_mode=0)
    alg.run()
    print('GA result:', alg.best_obj[0])
    alg = ACO(problem, num_pop=50, num_iter=100, show_mode=0)
    alg.run()
    print('ACO result:', alg.best_obj[0])
    alg = FI(problem)
    alg.run()
    print('FI result:', alg.best_obj[0])
    alg = GFLS(problem, num_iter=1000, show_mode=0)
    alg.run()
    print('GFLS result:', alg.best_obj[0])
    problem.plot_(alg.best)
