from choccy.algorithms.multi import *
from choccy.problems.multi.MOKP import *
from choccy.algorithms import Comparator

# 初始化问题
problem = MOKP(1000)
# 初始化算法集合
algorithms = dict()
n_sols, max_iter = 100, 100
algorithms['NNDREA'] = NNDREA(n_sols, max_iter)
algorithms['NSGA-II'] = NSGAII(n_sols, max_iter)
algorithms['SPEA2'] = SPEA2(n_sols, max_iter)
# 构建算法比较器
comparator = Comparator(problem, algorithms, visual_mode='obj')
# 设置追踪指标
comparator.track_metrics(['hv'])
# 运行算法比较器
comparator.run_comparison()
# 报告优化结果信息
comparator.report_result()
# 绘制收敛指标图像
comparator.plot('metric')

