from choccy.algorithms import Evaluator
from choccy.algorithms.single import *
from choccy.algorithms.multi import NNDREA
from choccy.problems.single import BinaryKP

# 创建问题列表集合
problems = [BinaryKP(n_vars=200)]
# 初始化算法集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 50, 1000
# 在算法字典集合中加入算法
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['DE'] = DE(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['PSO'] = PSO(n_sols, max_iter)
algorithms['DPKP'] = DPKP()
algorithms['GreedyKP'] = GreedyKP()
algorithms['NNDREA'] = NNDREA(n_sols, max_iter)
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
