import numpy as np
from choccy.algorithms.single import *
from choccy.problems.single import TSP
from choccy.utilities.handler import load_tsp_coord, load_tsp_matrix


tsp_data = load_tsp_coord('instance/eil51.tsp')
# tsp_data = load_tsp_matrix('instance/gr24.tsp')
# 初始化问题
problem = TSP(n_vars=tsp_data['dimension'],
              locations=tsp_data['node_coord'],
              dist_mat=tsp_data['dist_matrix'],
              round_dist=True)
# 初始化算法
algorithm = GuidedFastLocalSearch(zero_start=True)
algorithm.optimize(problem)
algorithm.report_result()
algorithm.plot('problem')