from choccy.algorithms.multi import *
from choccy.problems.multi.DTLZ import *
from choccy.algorithms import Evaluator

# 初始化问题集合
problems = dict()
problems['DTLZ1'] = DTLZ1()
problems['DTLZ2'] = DTLZ2()
# 初始化算法集合
algorithms = dict()
n_sols, max_iter = 100, 100
algorithms['NSGA-II'] = NSGAII(n_sols, max_iter)
algorithms['SPEA2'] = SPEA2(n_sols, max_iter)
algorithms['MOEAD'] = MOEAD(n_sols, max_iter)
# 构建问题-算法评估器
evaluator = Evaluator(problems, algorithms, n_runs=10)
# 设置追踪指标
evaluator.track_metrics(['igd', 'gd', 'hv'])
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
