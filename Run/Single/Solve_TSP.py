from Problems.Single.TSP import TSP
from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.ACO import ACO
from Algorithms.Single.HGA_TSP import HGATSP
from Algorithms.Single.FI import FI
from Algorithms.Single.GFLS import GFLS
from Algorithms.CONTRAST import CONTRAST
from Algorithms.ALGORITHM import ALGORITHM

if __name__ == '__main__':
    problem = TSP(30)
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(problem, num_pop, num_iter)
    algorithms['SA'] = SA(problem, num_pop, num_iter)
    algorithms['ACO'] = ACO(problem, num_pop, num_iter)
    algorithms['HGA-TSP'] = HGATSP(problem, num_pop, num_iter)
    algorithms['FI'] = FI(problem)
    algorithms['GFLS'] = GFLS(problem, num_iter)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.OBJ, same_init=True)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)
    algorithms['HGA-TSP'].plot(show_mode=ALGORITHM.PRB)
