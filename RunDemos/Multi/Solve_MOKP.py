from Problems.Multi import MOKP
from Algorithms import View, Comparator
from Algorithms.Multi import NNDREA, NSGAII, MOEAD, SPEA2
"""多个算法优化MOKP问题的对比测试"""

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
