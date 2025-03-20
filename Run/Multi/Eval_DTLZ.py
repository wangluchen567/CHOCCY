from Problems.Multi.DTLZ.DTLZ1 import DTLZ1
from Problems.Multi.DTLZ.DTLZ2 import DTLZ2
from Problems.Multi.DTLZ.DTLZ3 import DTLZ3
from Problems.Multi.DTLZ.DTLZ4 import DTLZ4
from Problems.Multi.DTLZ.DTLZ5 import DTLZ5
from Algorithms.Multi.NSGAII import NSGAII
from Algorithms.Multi.MOEAD import MOEAD
from Algorithms.Multi.SPEA2 import SPEA2
from Algorithms.Evaluator import Evaluator

if __name__ == '__main__':
    problems = [
        DTLZ1(),
        DTLZ2(),
        DTLZ3(),
        DTLZ4(),
        DTLZ5(),
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
    print('*** time ***')
    evaluator.prints('time')