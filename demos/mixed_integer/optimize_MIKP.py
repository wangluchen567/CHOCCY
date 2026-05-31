from choccy.algorithms.single import *
from MixedIntegerKnapsack import MixedIntegerKnapsack

# 初始化问题
problem = MixedIntegerKnapsack()
# 初始化算法
algorithm = DE(n_sols=50, max_iter=1000, visual_mode='log')
# 运行算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
# 绘制目标值图像
algorithm.plot('obj')
