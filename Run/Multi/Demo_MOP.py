from Problems.Multi import MOP1
from Algorithms.Multi import MOEAD
from Algorithms.Multi import NSGAII
from Algorithms.Multi import SPEA2


if __name__ == '__main__':
    problem = MOP1()
    algorithm = NSGAII(pop_size=100, max_iter=100, show_mode=1)
    algorithm.solve(problem)
