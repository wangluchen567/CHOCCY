from Problems.Multi import DTLZ1, DTLZ2, DTLZ3, DTLZ4
from Algorithms.Multi import NSGAII, MOEAD, SPEA2
from Algorithms import Evaluator

if __name__ == '__main__':
    problems = [
        DTLZ1(),
        DTLZ2(),
        DTLZ3(),
        DTLZ4()
    ]
    pop_size, max_iter = 100, 100
    algorithms = {
        "NSGA-II": NSGAII(pop_size, max_iter),
        "MOEA/D": MOEAD(pop_size, max_iter),
        "SPEA2": SPEA2(pop_size, max_iter),
    }
    evaluator = Evaluator(problems, algorithms, num_run=10, same_init=True)
    evaluator.run()
    # # 使用多核CPU并行优化
    # evaluator.run_parallel(num_processes=10)
    print('*** HV ***')
    evaluator.prints()
    print('*** GD ***')
    evaluator.prints('GD')
    print('*** IGD ***')
    evaluator.prints('IGD')
    print('*** time ***')
    evaluator.prints('time')