from Problems.Single import TSP
from Algorithms.Single import GA
from Algorithms.Single import SA
from Algorithms.Single import ACO
from Algorithms.Single import HGATSP
from Algorithms.Single import FI
from Algorithms.Single import GFLS
from Algorithms import View, Comparator

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
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ, same_init=True)
    comparator.run()
    comparator.plot(show_mode=View.OBJ)
    algorithms['HGA-TSP'].plot(show_mode=View.PROB)
