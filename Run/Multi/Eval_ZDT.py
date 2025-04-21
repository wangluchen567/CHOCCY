from Problems.Multi import ZDT1
from Problems.Multi import ZDT2
from Problems.Multi import ZDT3
from Problems.Multi import ZDT4
from Problems.Multi import ZDT5
from Problems.Multi import ZDT6
from Algorithms.Multi import NSGAII
from Algorithms.Multi import MOEAD
from Algorithms.Multi import SPEA2
from Algorithms import Evaluator

if __name__ == '__main__':
    problems = [
        ZDT1(),
        ZDT2(),
        ZDT3(),
        # ZDT4(),
        # ZDT5(),
        # ZDT6(),
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
    print('*** Time ***')
    evaluator.prints('time')
