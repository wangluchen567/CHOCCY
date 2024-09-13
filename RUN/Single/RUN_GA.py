from Algorithms.Single.GA import GA
from Problems.Single.KP import KP
from Problems.Single.TSP import TSP
from Problems.Single.Ackley import Ackley
from Problems.Single.Square import Square
from Problems.Single.FixLabelCluster import FixLabelCluster
from Problems.Single.MixFixLabelCluster import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=2)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=3)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


def Solve_Square():
    problem = Square(num_dec=2)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=3)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=1)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


def Solve_TSP():
    problem = TSP(30)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=4)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=4)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=4)
    alg.run()
    print(alg.get_best()[1])
    # alg.plot()
    alg.plot_scores()


if __name__ == '__main__':
    Solve_Ackley()
    # Solve_Square()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    # Solve_MixFixLabelCluster()
