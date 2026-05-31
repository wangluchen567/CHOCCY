from choccy.algorithms.single import Adam, DE
from choccy.problems.single import Classification

# 初始化问题
problem = Classification()
# 使用Adam算法求解
algorithm = Adam(learning_rate=0.5, visual_mode='prob')
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('prob')
# 使用差分进化算法求解
algorithm = DE(visual_mode='log')
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('prob')
