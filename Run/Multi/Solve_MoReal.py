from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.ZDT.ZDT2 import ZDT2
from Problems.Multi.ZDT.ZDT3 import ZDT3
from Problems.Multi.ZDT.ZDT4 import ZDT4
from Problems.Multi.ZDT.ZDT6 import ZDT6
from Problems.Multi.DTLZ.DTLZ1 import DTLZ1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Problems.Multi.DTLZ.DTLZ3 import DTLZ3
from Problems.Multi.DTLZ.DTLZ4 import DTLZ4
from Problems.Multi.DTLZ.DTLZ5 import DTLZ5
from Problems.Multi.DTLZ.DTLZ7 import DTLZ7
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.CONTRAST import CONTRAST


if __name__ == '__main__':
    problem = DTLZ1()
    algorithms = dict()
    num_pop, num_iter = 100, 300
    algorithms['NSGA-II'] = NSGAII(problem, num_pop, num_iter)
    algorithms['MOEA/D'] = MOEAD(problem, num_pop, num_iter)
    algorithms['SPEA2'] = SPEA2(problem, num_pop, num_iter)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.BAR, same_init=True)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.OBJ)

