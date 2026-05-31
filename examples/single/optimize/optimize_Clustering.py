from choccy.algorithms.single import *
from choccy.problems.single import Clustering

# 初始化问题
problem = Clustering(n_features=3)

# 使用差分进化算法求解
algorithm = DE()
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('obj')
algorithm.plot('prob')

# 使用遗传算法求解
algorithm = GA()
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('obj')
algorithm.plot('prob')

# 使用粒子群算法求解
algorithm = PSO()
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('obj')
algorithm.plot('prob')

# 使用模拟退火算法求解
algorithm = SA()
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('obj')
algorithm.plot('prob')