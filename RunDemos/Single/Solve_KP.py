import numpy as np
import seaborn as sns
from Problems.Single import KP
from Algorithms import View, Comparator
from Algorithms.Single import SA, GA, DPKP, NNDREAS, GreedyKP
"""多个算法优化KP问题的对比测试"""

if __name__ == '__main__':
    problem = KP(num_dec=100)
    algorithms = dict()
    algorithms['GA'] = GA(pop_size=100, max_iter=100)
    algorithms['SA'] = SA(pop_size=100, max_iter=100, perturb_prob=0.1)
    algorithms['DP'] = DPKP()
    algorithms['Greedy'] = GreedyKP()
    algorithms['NNDREA'] = NNDREAS(pop_size=100, max_iter=100)
    comparator = Comparator(problem, algorithms, show_mode=View.SCORE, same_init=True)
    comparator.run()
    print(np.min(algorithms['NNDREA'].pop_weights), np.max(algorithms['NNDREA'].pop_weights))
    sns.heatmap(algorithms['NNDREA'].pop_weights, cmap="YlGnBu")
