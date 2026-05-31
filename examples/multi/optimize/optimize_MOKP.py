import numpy as np
from choccy.algorithms.multi import *
from choccy.problems.multi.MOKP import *

# 设置随机种子（可选）
np.random.seed(42)
# 初始化问题
problem = MOKP(1000)
# 初始化算法
algorithm = NNDREA(n_sols=100, max_iter=100, visual_mode='obj')
# 设置追踪指标
algorithm.track_metrics(['hv'])
# 运行算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
# 绘制指标收敛图像
algorithm.plot('metric')
