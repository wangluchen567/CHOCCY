"""
SPEA2 算法测试
"""
import numpy as np
from choccy.algorithms.multi import SPEA2
from choccy.problems.multi import ZDT1


def test_spea2_produces_front_on_zdt1():
    """SPEA2 在 ZDT1 上应产生 Pareto 前沿"""
    np.random.seed(42)
    problem = ZDT1()
    algo = SPEA2(n_sols=100, max_iter=100, visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2
    assert best.objs.shape[1] == 2
