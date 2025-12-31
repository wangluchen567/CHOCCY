from Problems.Single import Ackley  # 导入问题
from Algorithms import View, Comparator  # 导入绘图参数与比较器
from Algorithms.Single import GA, SA, DE, PSO  # 导入多种算法
"""多个算法优化Ackley问题的对比测试"""

if __name__ == '__main__':
    # 定义要求解的问题
    problem = Ackley(num_dec=2)
    # 初始化算法字典集合
    algorithms = dict()
    # 定义算法的参数(参数统一)
    pop_size, max_iter = 100, 100
    # 在算法字典集合中加入算法
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['PSO'] = PSO(pop_size, max_iter)
    algorithms['DE/rand/1'] = DE(pop_size, max_iter, operator_type=DE.RAND_1)
    algorithms['DE/rand/2'] = DE(pop_size, max_iter, operator_type=DE.RAND_2)
    algorithms['DE/best/1'] = DE(pop_size, max_iter, operator_type=DE.BEST_1)
    algorithms['DE/best/2'] = DE(pop_size, max_iter, operator_type=DE.BEST_2)
    # 使用比较器比较算法，并展示各个算法目标值的效果比较
    comparator = Comparator(problem, algorithms, show_mode=View.OBJ, same_init=True)
    comparator.run()