from Problems.Single.TSP import TSP
from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.ACO import ACO
from Algorithms.Single.HGA_TSP import HGATSP
from Algorithms.Single.FI import FI
from Algorithms.Single.GFLS import GFLS

if __name__ == '__main__':
    problem = TSP(100)
    alg = GA(problem, num_pop=100, num_iter=1000, show_mode=0)
    alg.run()
    print('GA result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg = SA(problem, num_pop=10, num_iter=10000, show_mode=0)
    alg.run()
    print('SA result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg = ACO(problem, num_pop=50, num_iter=100, show_mode=0)
    alg.run()
    print('ACO result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg = HGATSP(problem, num_pop=100, num_iter=1000, show_mode=0)
    alg.run()
    print('HGA_TSP result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg = FI(problem)
    alg.run()
    print('FI result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg = GFLS(problem, num_iter=1000, show_mode=0)
    alg.run()
    print('GFLS result:', alg.best_obj[0])
    print("Run time: ", alg.run_time)
    alg.plot(show_mode=GFLS.PRB)
