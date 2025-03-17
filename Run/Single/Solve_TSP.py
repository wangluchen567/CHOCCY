from Problems.Single.TSP import TSP
from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.ACO import ACO
from Algorithms.Single.HGA_TSP import HGATSP
from Algorithms.Single.FI import FI
from Algorithms.Single.GFLS import GFLS
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Comparator import Comparator

if __name__ == '__main__':
    problem = TSP(30)
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(problem, num_pop, num_iter)
    algorithms['SA'] = SA(problem, num_pop, num_iter, perturb_prob=0.5)
    algorithms['ACO'] = ACO(problem, num_pop, num_iter)
    algorithms['HGA-TSP'] = HGATSP(problem, num_pop, num_iter)
    algorithms['FI'] = FI(problem)
    algorithms['GFLS'] = GFLS(problem, num_iter)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ, same_init=True)
    comparator.run_compare()
    comparator.plot(show_mode=Comparator.OBJ)
    algorithms['HGA-TSP'].plot(show_mode=ALGORITHM.PRB)
