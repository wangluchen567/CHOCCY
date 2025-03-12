from Problems.Multi.Practical.MOKP import MOKP
from Algorithms.Multi.NNDREA import NNDREA
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.CONTRAST import CONTRAST

if __name__ == '__main__':
    problem = MOKP(10000)
    algorithms = dict()
    algorithms['NSGAII'] = NSGAII(problem, num_pop=100, num_iter=100)
    algorithms['NNDREA'] = NNDREA(problem, num_pop=100, num_iter=100)
    contrast = CONTRAST(problem, algorithms, show_mode=1)
    contrast.run_contrast()