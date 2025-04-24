from Algorithms import View
from Algorithms.Multi import NNDREA
from Problems.Multi import MOKP


if __name__ == '__main__':
    problem = MOKP(10000)
    algorithm = NNDREA(pop_size=100, max_iter=100, show_mode=View.OBJ)
    algorithm.solve(problem)
    print("HV: ", algorithm.cal_score('HV'))
    print("time(s): ", algorithm.run_time)
    algorithm.plot(show_mode=1)
    algorithm.plot_scores()
