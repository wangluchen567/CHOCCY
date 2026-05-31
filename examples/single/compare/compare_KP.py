import numpy as np
from choccy.algorithms.single import *
from choccy.algorithms.multi import NNDREA
from choccy.algorithms import Comparator
from choccy.problems.single import BinaryKP

# 设置随机种子（可选）
np.random.seed(42)
# 初始化问题
problem = BinaryKP()
# 初始化算法集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 50, 1000
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['DE'] = DE(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['PSO'] = PSO(n_sols, max_iter)
algorithms['DPKP'] = DPKP()
algorithms['Greedy'] = GreedyKP()
algorithms['NNDREA'] = NNDREA(n_sols, max_iter)
# 构建算法比较器
comparator = Comparator(problem, algorithms, same_start=True, visual_mode='log')
# 运行算法比较器
comparator.run_comparison()
# 报告优化结果信息
comparator.report_result()
# 绘制目标值图像
comparator.plot('obj')
