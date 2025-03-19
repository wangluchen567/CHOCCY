from Algorithms.Single.GA import GA
from Algorithms.Single.SA import SA
from Algorithms.Single.DE import DE
from Algorithms.Single.PSO import PSO
from Problems.Single.Ackley import Ackley
from Algorithms.Evaluator import Evaluator

if __name__ == '__main__':
    problems = [Ackley(num_dec=30)]
    algorithms = dict()
    num_pop, num_iter = 100, 100
    algorithms['GA'] = GA(num_pop, num_iter)
    algorithms['SA'] = SA(num_pop, num_iter)
    algorithms['PSO'] = PSO(num_pop, num_iter)
    algorithms['DE/rand/1'] = DE(num_pop, num_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(num_pop, num_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(num_pop, num_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(num_pop, num_iter, operator_type=DE.BEST2)
    evaluator = Evaluator(problems, algorithms, num_run=30, same_init=True)
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
