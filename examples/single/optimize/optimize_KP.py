import numpy as np
from choccy.algorithms.single import *
from choccy.algorithms.multi import NNDREA
from choccy.problems.single import BinaryKP

# 设置随机种子（可选）
np.random.seed(42)
# 创建0-1背包问题
problem = BinaryKP(n_vars=200)
# 使用动态规划算法求解
algorithm = DPKP()
algorithm.optimize(problem)
algorithm.report_result()
# 使用贪心算法求解
algorithm = GreedyKP()
algorithm.optimize(problem)
algorithm.report_result()
# 使用遗传算法求解
algorithm = GA(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
# 使用模拟退火算法求解
algorithm = SA(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
# 使用粒子群算法求解
algorithm = PSO(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
# 使用差分进化算法求解
algorithm = DE(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
# 使用 NNDREA 求解
algorithm = NNDREA(n_sols=50, max_iter=1000)
algorithm.optimize(problem)
algorithm.report_result()
