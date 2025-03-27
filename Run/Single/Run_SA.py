from Algorithms.Single import SA
from Problems.Single import KP
from Problems.Single import TSP
from Problems.Single import Ackley
from Problems.Single import Square
from Problems.Single import FixLabelCluster
from Problems.Single import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=2)
    algorithm = SA(pop_size=1, max_iter=10000, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_Square():
    problem = Square(num_dec=2)
    algorithm = SA(pop_size=1, max_iter=10000, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    algorithm = SA(pop_size=1, max_iter=10000, perturb_prob=1 / 100, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_TSP():
    problem = TSP(30)
    algorithm = SA(pop_size=1, max_iter=10000, perturb_prob=0.5, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=SA.PRB)


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    algorithm = SA(pop_size=1, max_iter=10000, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=SA.PRB)


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    algorithm = SA(pop_size=1, max_iter=10000, show_mode=0)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=SA.PRB)


if __name__ == '__main__':
    # Solve_Ackley()
    # Solve_Square()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    Solve_MixFixLabelCluster()

