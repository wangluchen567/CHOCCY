from Algorithms.Single import GA
from Algorithms.Single import SA
from Algorithms.Single import DE
from Algorithms.Single import PSO
from Problems.Single import Ackley
from Algorithms import View, Comparator

if __name__ == '__main__':
    problem = Ackley(num_dec=2)
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['PSO'] = PSO(pop_size, max_iter)
    algorithms['DE/rand/1'] = DE(pop_size, max_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(pop_size, max_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(pop_size, max_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(pop_size, max_iter, operator_type=DE.BEST2)
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ, same_init=True)
    comparator.run()
    comparator.plot(show_mode=View.OBJ)
