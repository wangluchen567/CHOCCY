import numpy as np
from choccy.algorithms.single import *
from choccy.problems.single import FixedSizeCluster

np.random.seed(42)
# 初始化问题
problem = FixedSizeCluster()
# 使用遗传算法求解
algorithm = GA(visual_mode='prob')
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('prob')
# 使用模拟退火算法求解
algorithm = SA(perturb_rate=0.5)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('prob')
