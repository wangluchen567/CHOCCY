import numpy as np
from choccy.algorithms.single import *
from choccy.algorithms import Comparator
from choccy.problems.single import Ackley


# 设置随机种子（可选）
np.random.seed(42)
# 初始化问题
problem = Ackley(n_vars=10)
# 初始化算法集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 50, 200
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['PSO'] = PSO(n_sols, max_iter)
algorithms['DE/rand/1'] = DE(n_sols, max_iter, operator_type=DE.RAND_1)
algorithms['DE/rand/2'] = DE(n_sols, max_iter, operator_type=DE.RAND_2, cross_probs=0.3)
algorithms['DE/best/1'] = DE(n_sols, max_iter, operator_type=DE.BEST_1, cross_probs=0.3)
algorithms['DE/best/2'] = DE(n_sols, max_iter, operator_type=DE.BEST_2)
# 构建算法比较器
comparator = Comparator(problem, algorithms, same_start=True, visual_mode='obj')
# 运行算法比较器
comparator.run_comparison()
# 报告优化结果信息
comparator.report_result()
# 绘制目标值图像
comparator.plot('obj')
