from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NNDREA import NNDREA
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.Comparator import Comparator

if __name__ == '__main__':
    problem = MOKP(10000)
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(pop_size, max_iter)
    algorithms['MOEA/D'] = MOEAD(pop_size, max_iter)
    algorithms['SPEA2'] = SPEA2(pop_size, max_iter)
    algorithms['NNDREA'] = NNDREA(pop_size, max_iter)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ)
    comparator.run()
    comparator.plot(show_mode=Comparator.SCORE)
