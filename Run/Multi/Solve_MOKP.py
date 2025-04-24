from Problems.Multi import MOKP
from Algorithms.Multi import NNDREA
from Algorithms.Multi import NSGAII
from Algorithms.Multi import MOEAD
from Algorithms.Multi import SPEA2
from Algorithms import View, Comparator

if __name__ == '__main__':
    problem = MOKP(10000)
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(pop_size, max_iter)
    algorithms['MOEA/D'] = MOEAD(pop_size, max_iter)
    algorithms['SPEA2'] = SPEA2(pop_size, max_iter)
    algorithms['NNDREA'] = NNDREA(pop_size, max_iter)
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ)
    comparator.run()
    comparator.plot(show_mode=View.SCORE)
