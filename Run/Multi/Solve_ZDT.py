from Problems.Multi import ZDT1
from Algorithms.Comparator import View, Comparator
from Algorithms.Multi import MOEAD, NSGAII, SPEA2

if __name__ == '__main__':
    problem = ZDT1()
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['NSGA-II'] = NSGAII(pop_size, max_iter)
    algorithms['MOEA/D'] = MOEAD(pop_size, max_iter)
    algorithms['SPEA2'] = SPEA2(pop_size, max_iter)
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ, same_init=True)
    comparator.set_score_type('IGD')  # 设置评价指标为IGD
    comparator.run()
    comparator.plot(show_mode=View.SCORE)

