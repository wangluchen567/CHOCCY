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
    pop_size, max_iter = 100, 100
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter, perturb_prob=0.5)
    algorithms['ACO'] = ACO(pop_size, max_iter)
    algorithms['HGA-TSP'] = HGATSP(pop_size, max_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(max_iter)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ, same_init=True)
    comparator.run()
    comparator.plot(show_mode=Comparator.OBJ)
    algorithms['HGA-TSP'].plot(show_mode=ALGORITHM.PRB)
