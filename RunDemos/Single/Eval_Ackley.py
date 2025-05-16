from Problems.Single import Ackley
from Algorithms import Evaluator
from Algorithms.Single import GA, SA, DE, PSO
"""多种算法优化Ackley问题的评估测试"""

if __name__ == '__main__':
    problems = [Ackley(num_dec=30)]
    algorithms = dict()
    pop_size, max_iter = 100, 100
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['PSO'] = PSO(pop_size, max_iter)
    algorithms['DE/rand/1'] = DE(pop_size, max_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(pop_size, max_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(pop_size, max_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(pop_size, max_iter, operator_type=DE.BEST2)
    evaluator = Evaluator(problems, algorithms, num_run=30, same_init=True)
    evaluator.run()
    # # 使用多核CPU并行优化
    # evaluator.run_parallel(num_processes=10)
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
