from Algorithms.Single.GA import GA
from Problems.Single.KP import KP
from Problems.Single.TSP import TSP
from Problems.Single.Ackley import Ackley
from Problems.Single.Square import Square
from Problems.Single.FixLabelCluster import FixLabelCluster
from Problems.Single.MixFixLabelCluster import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=2)
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.OAD3)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_Square():
    problem = Square(num_dec=2)
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.OAD2)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.OBJ)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()


def Solve_TSP():
    problem = TSP(30)
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.PRB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PRB)


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.PRB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PRB)


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    algorithm = GA(num_pop=100, num_iter=100, show_mode=GA.PRB)
    algorithm.solve(problem)
    best, best_obj, best_con = algorithm.get_best()
    print(best_obj)
    algorithm.plot_scores()
    algorithm.plot(show_mode=GA.PRB)


if __name__ == '__main__':
    # Solve_Ackley()
    # Solve_Square()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    Solve_MixFixLabelCluster()
