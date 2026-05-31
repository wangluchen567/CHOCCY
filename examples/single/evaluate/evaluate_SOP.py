from choccy.problems.single import *
from choccy.algorithms.single import *
from choccy.algorithms import Evaluator

# 创建问题列表集合
problems = [SOP1(), SOP5(), SOP10()]
# 初始化算法字典集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 50, 1000
# 在算法字典集合中加入算法
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['PSO'] = PSO(n_sols, max_iter, c1=1.2, c2=1.8)
algorithms['DE/rand/1'] = DE(n_sols, max_iter, operator_type=DE.RAND_1)
algorithms['DE/rand/2'] = DE(n_sols, max_iter, operator_type=DE.RAND_2, cross_probs=0.3)
algorithms['DE/best/1'] = DE(n_sols, max_iter, operator_type=DE.BEST_1, cross_probs=0.3)
algorithms['DE/best/2'] = DE(n_sols, max_iter, operator_type=DE.BEST_2)
# 构建问题-算法评估器
evaluator = Evaluator(problems, algorithms, n_runs=10, same_start=True)
# 运行问题-算法评估器
evaluator.run_evaluation()
# 报告优化结果信息
evaluator.report_result()
# 绘制小提琴图
evaluator.plot_violin()
# 绘制箱线图
evaluator.plot_box()
# 绘制核密度估计图
evaluator.plot_kde()
