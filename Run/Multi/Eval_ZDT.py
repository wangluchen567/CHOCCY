from Problems.Multi.ZDT.ZDT1 import ZDT1
from Problems.Multi.ZDT.ZDT2 import ZDT2
from Problems.Multi.ZDT.ZDT3 import ZDT3
from Problems.Multi.ZDT.ZDT4 import ZDT4
from Problems.Multi.ZDT.ZDT5 import ZDT5
from Problems.Multi.ZDT.ZDT6 import ZDT6
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.Evaluator import Evaluator

if __name__ == '__main__':
    problems = [
        ZDT1(),
        ZDT2(),
        ZDT3(),
        ZDT4(),
        ZDT5(),
        ZDT6(),
    ]
    pop_size, max_iter = 100, 100
    algorithms = {
        "NSGA-II": NSGAII(pop_size, max_iter),
        "MOEA/D": MOEAD(pop_size, max_iter),
        "SPEA2": SPEA2(pop_size, max_iter),
    }
    evaluator = Evaluator(problems, algorithms, num_run=10, same_init=True)
    evaluator.run()
    print('*** HV ***')
    evaluator.prints()
    print('*** GD ***')
    evaluator.prints('GD')
    print('*** IGD ***')
    evaluator.prints('IGD')
    print('*** Time ***')
    evaluator.prints('time')
