from Algorithms.Single.SA import SA
from Problems.Single.KP import KP
from Problems.Single.TSP import TSP
from Problems.Single.Ackley import Ackley
from Problems.Single.Square import Square
from Problems.Single.FixLabelCluster import FixLabelCluster
from Problems.Single.MixFixLabelCluster import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=2)
    alg = SA(problem, num_pop=1, num_iter=10000, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()


def Solve_Square():
    problem = Square(num_dec=2)
    alg = SA(problem, num_pop=1, num_iter=10000, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    alg = SA(problem, num_pop=1, num_iter=10000, perturb_prob=1 / 100, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()


def Solve_TSP():
    problem = TSP(20)
    alg = SA(problem, num_pop=1, num_iter=10000, perturb_prob=1.0, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()
    alg.plot(show_mode=SA.PRB)


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    alg = SA(problem, num_pop=1, num_iter=10000, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()
    alg.plot(show_mode=SA.PRB)


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    alg = SA(problem, num_pop=1, num_iter=10000, show_mode=0)
    alg.run()
    print(alg.best_obj[0])
    # alg.plot()
    alg.plot_scores()
    alg.plot(show_mode=SA.PRB)


if __name__ == '__main__':
    # Solve_Ackley()
    # Solve_Square()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    Solve_MixFixLabelCluster()

