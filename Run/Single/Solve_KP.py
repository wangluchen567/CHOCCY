import numpy as np
import seaborn as sns
from Problems.Single.KP import KP
from Algorithms.Single.SA import SA
from Algorithms.Single.GA import GA
from Algorithms.Single.DP_KP import DPKP
from Algorithms.Single.NNDREAS import NNDREAS
from Algorithms.Single.Greedy_KP import GreedyKP
from Algorithms.CONTRAST import CONTRAST

if __name__ == '__main__':
    problem = KP(num_dec=100)
    algorithms = dict()
    algorithms['GA'] = GA(problem, num_pop=100, num_iter=100)
    algorithms['SA'] = SA(problem, num_pop=100, num_iter=100)
    algorithms['DP'] = DPKP(problem)
    algorithms['Greedy'] = GreedyKP(problem)
    algorithms['NNDREA'] = NNDREAS(problem, num_pop=100, num_iter=100)
    contrast = CONTRAST(problem, algorithms, show_mode=CONTRAST.BAR, same_init=True)
    contrast.run_contrast()
    contrast.plot(show_mode=CONTRAST.SCORE)
    print(np.min(algorithms['NNDREA'].pop_weights), np.max(algorithms['NNDREA'].pop_weights))
    sns.heatmap(algorithms['NNDREA'].pop_weights, cmap="YlGnBu")
