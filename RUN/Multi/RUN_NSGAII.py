from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.ZDT.ZDT2 import ZDT2
from Problems.Multi.ZDT.ZDT3 import ZDT3
from Problems.Multi.ZDT.ZDT4 import ZDT4
from Problems.Multi.ZDT.ZDT6 import ZDT6
from Problems.Multi.DTLZ.DTLZ1 import DTLZ1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Problems.Multi.DTLZ.DTLZ3 import DTLZ3
from Problems.Multi.DTLZ.DTLZ4 import DTLZ4
from Problems.Multi.DTLZ.DTLZ5 import DTLZ5
from Problems.Multi.DTLZ.DTLZ7 import DTLZ7
from Algorithms.Multi.NSGAII import NSGAII

if __name__ == '__main__':
    problem = ZDT3()
    alg = NSGAII(problem, num_pop=100, num_iter=100, show_mode=1)
    alg.run()
    print("Run time: ", alg.run_time)
    alg.plot(show_mode=1)
    print("Final score: ", alg.get_scores()[-1])
    alg.plot_scores()

