from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.DE import DE
from Algorithms.CONTRAST import CONTRAST
from Problems.Single.Ackley import Ackley

if __name__ == '__main__':
    problem = Ackley(num_dec=2)
    algorithms = dict()
    algorithms['GA'] = GA(problem, num_pop=100, num_iter=100)
    algorithms['SA'] = SA(problem, num_pop=100, num_iter=100)
    algorithms['DE/rand/1'] = DE(problem, num_pop=100, num_iter=100, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(problem, num_pop=100, num_iter=100, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(problem, num_pop=100, num_iter=100, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(problem, num_pop=100, num_iter=100, operator_type=DE.BEST2)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.BAR, same_init=True)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)