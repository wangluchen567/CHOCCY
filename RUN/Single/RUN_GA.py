from Algorithms.Single.GA import GA
from Problems.Single.KP import KP
from Problems.Single.TSP import TSP
from Problems.Single.Ackley import Ackley
from Problems.Single.Square import Square
from Problems.Single.FixLabelCluster import FixLabelCluster
from Problems.Single.MixFixLabelCluster import MixFixLabelCluster


def Solve_Ackley():
    problem = Ackley(num_dec=1)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.OAD)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()


def Solve_Square():
    problem = Square(num_dec=2)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.OAD)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()


def Solve_KP():
    problem = KP(num_dec=100)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.OBJ)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()


def Solve_TSP():
    problem = TSP(30)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.PRB)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()
    alg.plot(show_mode=4)


def Solve_FixLabelCluster():
    problem = FixLabelCluster()
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.PRB)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()
    alg.plot(show_mode=4)


def Solve_MixFixLabelCluster():
    problem = MixFixLabelCluster(90)
    alg = GA(problem, num_pop=100, num_iter=100, show_mode=GA.PRB)
    alg.run()
    print(alg.best_obj[0])
    alg.plot_scores()
    alg.plot(show_mode=4)


if __name__ == '__main__':
    # Solve_Ackley()
    # Solve_Square()
    # Solve_KP()
    # Solve_TSP()
    # Solve_FixLabelCluster()
    Solve_MixFixLabelCluster()
