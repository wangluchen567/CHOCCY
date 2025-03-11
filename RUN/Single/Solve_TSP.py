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
    algorithms['GA'] = GA(problem, num_pop=100, num_iter=100)
    algorithms['SA'] = SA(problem, num_pop=100, num_iter=100)
    algorithms['ACO'] = ACO(problem, num_pop=100, num_iter=100)
    algorithms['HGA-TSP'] = HGATSP(problem, num_pop=100, num_iter=100)
    algorithms['FI'] = FI(problem)
    algorithms['GFLS'] = GFLS(problem, num_iter=100)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.BAR)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)
    algorithms['HGA-TSP'].plot(show_mode=ALGORITHM.PRB)
