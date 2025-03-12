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
from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.NSGAII_DE import NSGAIIDE
from Algorithms.CONTRAST import CONTRAST


if __name__ == '__main__':
    problem = ZDT1()
    algorithms = dict()
    algorithms['NSGAII'] = NSGAII(problem, num_pop=100, num_iter=100)
    algorithms['MOEAD'] = MOEAD(problem, num_pop=100, num_iter=100)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.OBJ)
    contrast.run_contrast()
    contrast.plot(show_mode=1)

