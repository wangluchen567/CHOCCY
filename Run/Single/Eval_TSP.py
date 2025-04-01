from Problems.Single import TSP
from Algorithms.Single import GA
from Algorithms.Single import SA
from Algorithms.Single import ACO
from Algorithms.Single import HGATSP
from Algorithms.Single import FI
from Algorithms.Single import GFLS
from Algorithms import Evaluator

if __name__ == '__main__':
    problems = [TSP(30)]
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['ACO'] = ACO(pop_size, max_iter)
    algorithms['HGA-TSP'] = HGATSP(pop_size, max_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(max_iter)
    evaluator = Evaluator(problems, algorithms, num_run=10, same_init=True)
    evaluator.run()
    # 打印结果对比
    print('*** Obj ***')
    evaluator.prints()
    # 绘制小提琴图(设置为准确绘制)
    evaluator.plot_violin(cut=0)
    # 绘制箱线图
    evaluator.plot_box()
    # 打印时间对比
    print('*** Time ***')
    evaluator.prints('time')