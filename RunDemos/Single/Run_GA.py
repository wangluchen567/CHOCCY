from Algorithms.Single import GA
from Problems.Single import KP
from Problems.Single import TSP
from Problems.Single import Ackley
from Problems.Single import Sphere
from Problems.Single import FixLabelCluster
from Problems.Single import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=2)
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.MIX3D)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_Sphere():
    problem = Sphere(num_dec=2)
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.MIX2D)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.OBJ)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_TSP():
    problem = TSP(30)
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.PROB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PROB)


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.PROB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PROB)


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    algorithm = GA(pop_size=100, max_iter=100, show_mode=GA.PROB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PROB)


if __name__ == '__main__':
    # Solve_Ackley()
    # Solve_Sphere()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    Solve_MixFixLabelCluster()
