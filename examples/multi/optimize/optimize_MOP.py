import numpy as np
from choccy.algorithms.multi import *
from choccy.problems.multi.MOP import *

# 设置随机种子（可选）
np.random.seed(42)
# 初始化问题
problem = MOP3()
# 初始化算法
algorithm = NSGAII(n_sols=100, max_iter=100, visual_mode='log')
# 设置日志间隔
algorithm.set_logger(log_interval=20)
# 运行算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
# 绘制目标值图像
algorithm.plot('obj')
