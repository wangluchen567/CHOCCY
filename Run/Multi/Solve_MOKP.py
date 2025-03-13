from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NNDREA import NNDREA
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.CONTRAST import CONTRAST

if __name__ == '__main__':
    problem = MOKP(10000)
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(problem, num_pop, num_iter)
    algorithms['MOEA/D'] = MOEAD(problem, num_pop, num_iter)
    algorithms['SPEA2'] = SPEA2(problem, num_pop, num_iter)
    algorithms['NNDREA'] = NNDREA(problem, num_pop, num_iter)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.OBJ)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)