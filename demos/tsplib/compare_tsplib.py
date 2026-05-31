from choccy.algorithms.single import *
from choccy.algorithms import Comparator
from choccy.problems.single import TSP
from choccy.utilities.handler import load_tsp_coord, load_tsp_matrix


tsp_data = load_tsp_coord('instance/eil51.tsp')
# tsp_data = load_tsp_matrix('instance/gr24.tsp')
# 初始化问题
problem = TSP(n_vars=tsp_data['dimension'],
              locations=tsp_data['node_coord'],
              dist_mat=tsp_data['dist_matrix'],
              round_dist=True)

# 初始化算法集合
algorithms = dict()
# 定义算法的参数(参数统一)
n_sols, max_iter = 100, 1000
algorithms['GA'] = GA(n_sols, max_iter)
algorithms['SA'] = SA(n_sols, max_iter)
algorithms['ACO'] = ACO(n_sols, max_iter)
algorithms['HGA-TSP'] = HGATSP(n_sols, max_iter)
algorithms['FarthestInsertion'] = FarthestInsertion()
algorithms['LocalSearch'] = LocalSearch()
algorithms['GuidedFastLocalSearch'] = GuidedFastLocalSearch(max_iter)
# 构建算法比较器
comparator = Comparator(problem, algorithms)
# 运行算法比较器
comparator.run_comparison()
# 报告优化结果信息
comparator.report_result()
# 绘制目标值图像
comparator.plot('obj')