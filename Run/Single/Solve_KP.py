import numpy as np
import seaborn as sns
from Problems.Single import KP
from Algorithms.Single import SA
from Algorithms.Single import GA
from Algorithms.Single import DPKP
from Algorithms.Single import NNDREAS
from Algorithms.Single import GreedyKP
from Algorithms import View, Comparator

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
    comparator.plot(show_mode=View.SCORE)
    print(np.min(algorithms['NNDREA'].pop_weights), np.max(algorithms['NNDREA'].pop_weights))
    sns.heatmap(algorithms['NNDREA'].pop_weights, cmap="YlGnBu")
