"""
MOEA/D 算法测试
"""
import numpy as np
from choccy.algorithms.multi import MOEAD
from choccy.problems.multi import ZDT1


def test_moead_produces_front_on_zdt1():
    """MOEA/D 在 ZDT1 上应产生 Pareto 前沿"""
    np.random.seed(42)
    problem = ZDT1()
    algo = MOEAD(n_sols=100, max_iter=100, visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2
    assert best.objs.shape[1] == 2
    assert np.all(best.objs >= 0)


def test_moead_with_tchebycheff():
    """MOEA/D 使用切比雪夫聚合应能正常运行"""
    np.random.seed(42)
    problem = ZDT1()
    algo = MOEAD(n_sols=100, max_iter=100, agg_method='tch', visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2


def test_moead_with_weightsum():
    """MOEA/D 使用加权和聚合应能正常运行"""
    np.random.seed(42)
    problem = ZDT1()
    algo = MOEAD(n_sols=100, max_iter=100, agg_method='wsm', visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2
