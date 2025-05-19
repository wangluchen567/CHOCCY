from Algorithms import Evaluator  # 导入评估器
from Problems.Single import SOP1, SOP5, SOP10  # 导入多个问题
from Algorithms.Single import GA, SA, DE, PSO  # 导入多种算法
"""多种算法优化多个SOP问题的评估测试"""

if __name__ == '__main__':
    # 创建要比较的多个问题列表集合
    problems = [SOP1(), SOP5(), SOP10()]
    # 初始化算法字典集合
    algorithms = dict()
    # 定义算法的参数(参数统一)
    pop_size, max_iter = 100, 100
    # 在算法字典集合中加入算法
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['PSO'] = PSO(pop_size, max_iter)
    algorithms['DE/rand/1'] = DE(pop_size, max_iter, operator_type=DE.RAND1)
    algorithms['DE/rand/2'] = DE(pop_size, max_iter, operator_type=DE.RAND2)
    algorithms['DE/best/1'] = DE(pop_size, max_iter, operator_type=DE.BEST1)
    algorithms['DE/best/2'] = DE(pop_size, max_iter, operator_type=DE.BEST2)
    # 使用评估器评估多种算法在多个问题上的表现
    evaluator = Evaluator(problems, algorithms, num_run=30, same_init=True)
    evaluator.run()
    # 打印结果对比
    evaluator.prints()
    # 绘制小提琴图
    evaluator.plot("violin")
    # 绘制箱线图
    evaluator.plot("box")
    # 绘制核密度估计图
    evaluator.plot("kde")
