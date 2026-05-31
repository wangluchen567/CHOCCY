import numpy as np
from choccy.problems.single import TSP
from choccy.algorithms.single import *

# 设置随机种子（可选）
np.random.seed(42)
# 创建旅行商问题
problem = TSP(n_vars=50)
# 使用遗传算法求解
algorithm = GA(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用模拟退火算法求解
algorithm = SA(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用蚁群算法求解
algorithm = ACO(n_sols=50, max_iter=200)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用混合遗传算法求解
algorithm = HGATSP(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用最远插入算法求解
algorithm = FarthestInsertion(zero_start=True)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用局部搜索算法(2-opt)求解
algorithm = LocalSearch(zero_start=True)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
# 使用引导式快速局部搜索算法求解
algorithm = GuidedFastLocalSearch(zero_start=True)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')
