from choccy.problems.single import *
from choccy.algorithms.single import DE

# 初始化问题
problem = SOP9(n_vars=2)
# 初始化算法（设置为决策空间与目标空间混合绘制）
algorithm = DE(n_sols=50, max_iter=100, visual_mode='h2d')
# 设置视角固定
algorithm.set_visual_config(fixed=True)
# 运行算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
# 绘制目标值图像
algorithm.plot('obj')
