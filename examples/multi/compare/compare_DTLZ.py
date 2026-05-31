from choccy.algorithms.multi import *
from choccy.problems.multi.DTLZ import *
from choccy.algorithms import Comparator

# 初始化问题
problem = DTLZ2()
# 初始化算法集合
algorithms = dict()
n_sols, max_iter = 100, 100
algorithms['NSGA-II'] = NSGAII(n_sols, max_iter)
algorithms['SPEA2'] = SPEA2(n_sols, max_iter)
algorithms['MOEAD'] = MOEAD(n_sols, max_iter)
# 构建算法比较器
comparator = Comparator(problem, algorithms, visual_mode='log')
# 设置追踪指标
comparator.track_metrics(['igd', 'gd', 'hv'])
# 运行算法比较器
comparator.run_comparison()
# 报告优化结果信息
comparator.report_result()
# 绘制目标空间图像
comparator.plot('obj')
