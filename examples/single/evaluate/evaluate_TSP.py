from choccy.algorithms import Evaluator
from choccy.algorithms.single import *
from choccy.problems.single import TSP

# 创建问题列表集合
problems = [TSP(30)]
# 初始化算法集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 50, 200
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['ACO'] = ACO(n_sols, max_iter)
algorithms['HGA-TSP'] = HGATSP(n_sols, max_iter)
algorithms['FI'] = FarthestInsertion()
algorithms['LS'] = LocalSearch()
algorithms['GFLS'] = GuidedFastLocalSearch(max_iter)
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
