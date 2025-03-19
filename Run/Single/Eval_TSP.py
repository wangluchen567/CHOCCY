from Problems.Single.TSP import TSP
from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.ACO import ACO
from Algorithms.Single.HGA_TSP import HGATSP
from Algorithms.Single.FI import FI
from Algorithms.Single.GFLS import GFLS
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Evaluator import Evaluator

if __name__ == '__main__':
    problems = [TSP(30)]
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(num_pop, num_iter)
    algorithms['SA'] = SA(num_pop, num_iter, perturb_prob=0.5)
    algorithms['ACO'] = ACO(num_pop, num_iter)
    algorithms['HGA-TSP'] = HGATSP(num_pop, num_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(num_iter)
    evaluator = Evaluator(problems, algorithms, num_run=10, same_init=True)
    evaluator.run()
    # 打印结果对比
    print('*** Obj ***')
    evaluator.prints()
    # 绘制小提琴图
    evaluator.plot_violin()
    # 绘制箱线图
    evaluator.plot_box()
    # 绘制核密度估计图
    evaluator.plot_kde()
    # 打印时间对比
    print('*** Time ***')
    evaluator.prints('time')