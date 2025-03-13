from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.DE import DE
from Algorithms.Single.PSO import PSO
from Algorithms.CONTRAST import CONTRAST
from Problems.Single.Ackley import Ackley

if __name__ == '__main__':
    problem = Ackley(num_dec=2)
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(problem, num_pop, num_iter)
    algorithms['SA'] = SA(problem, num_pop, num_iter)
    algorithms['PSO'] = PSO(problem, num_pop, num_iter)
    algorithms['DE/rand/1'] = DE(problem, num_pop, num_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(problem, num_pop, num_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(problem, num_pop, num_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(problem, num_pop, num_iter, operator_type=DE.BEST2)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.OBJ, same_init=True)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)