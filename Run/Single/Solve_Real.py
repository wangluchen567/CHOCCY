from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.DE import DE
from Algorithms.Single.PSO import PSO
from Problems.Single.Ackley import Ackley
from Algorithms.Comparator import Comparator

if __name__ == '__main__':
    problem = Ackley(num_dec=2)
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(num_pop, num_iter)
    algorithms['SA'] = SA(num_pop, num_iter)
    algorithms['PSO'] = PSO(num_pop, num_iter)
    algorithms['DE/rand/1'] = DE(num_pop, num_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(num_pop, num_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(num_pop, num_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(num_pop, num_iter, operator_type=DE.BEST2)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ, same_init=True)
    comparator.run()
    comparator.plot(show_mode=Comparator.OBJ)
