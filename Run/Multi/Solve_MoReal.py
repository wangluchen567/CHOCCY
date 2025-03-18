from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.Comparator import Comparator

if __name__ == '__main__':
    problem = ZDT1()
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(problem, num_pop, num_iter)
    algorithms['MOEA/D'] = MOEAD(problem, num_pop, num_iter)
    algorithms['SPEA2'] = SPEA2(problem, num_pop, num_iter)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ, same_init=True)
    comparator.set_score_type('IGD')  # 设置评价指标为IGD
    comparator.run_compare()
    comparator.plot(show_mode=Comparator.SCORE)

