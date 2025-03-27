from Problems.Multi import ZDT1
from Algorithms.Multi import MOEAD
from Algorithms.Multi import NSGAII
from Algorithms.Multi import SPEA2
from Algorithms.Comparator import Comparator

if __name__ == '__main__':
    problem = ZDT1()
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(pop_size, max_iter)
    algorithms['MOEA/D'] = MOEAD(pop_size, max_iter)
    algorithms['SPEA2'] = SPEA2(pop_size, max_iter)
    comparator = Comparator(problem, algorithms, show_mode=Comparator.OBJ, same_init=True)
    comparator.set_score_type('IGD')  # 设置评价指标为IGD
    comparator.run()
    comparator.plot(show_mode=Comparator.SCORE)

