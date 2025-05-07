from Problems.Single import TSP
from Algorithms import View, Comparator
from Algorithms.Single import GA, SA, ACO, HGATSP, FI, GFLS

if __name__ == '__main__':
    problem = TSP(30)
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['ACO'] = ACO(pop_size, max_iter)
    algorithms['HGA-TSP'] = HGATSP(pop_size, max_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(max_iter)
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ, same_init=True)
    comparator.run()
    algorithms['GFLS'].plot(show_mode=View.PROB)
