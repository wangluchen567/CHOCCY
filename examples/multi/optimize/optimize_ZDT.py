import numpy as np
from choccy.algorithms.multi import *
from choccy.problems.multi.ZDT import *

# 设置随机种子（可选）
np.random.seed(42)
# 初始化问题
problem = ZDT3()
# 初始化算法
algorithm = NSGAII(n_sols=100, max_iter=100, visual_mode='obj')
# 设置追踪指标
algorithm.track_metrics(['hv'])
# 运行算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
# 绘制指标收敛图像
algorithm.plot('metric')
# 保存最优解
algorithm.save_best('best')
