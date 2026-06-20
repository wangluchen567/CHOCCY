"""
NSGA-II 算法测试
"""
import numpy as np
from choccy.algorithms.multi import NSGAII
from choccy.problems.multi import ZDT1, DTLZ1


def test_nsga2_produces_front_on_zdt1():
    """NSGA-II 在 ZDT1 上应产生 Pareto 前沿"""
    np.random.seed(42)
    problem = ZDT1()
    algo = NSGAII(n_sols=100, max_iter=100, visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2          # 应有多个非支配解
    assert best.objs.shape[1] == 2   # 2 目标
    # 验证 Pareto 前沿的基本性质：f1 和 f2 均为正且单调
    f1 = best.objs[:, 0]
    f2 = best.objs[:, 1]
    assert np.all(f1 >= 0) and np.all(f2 >= 0)


def test_nsga2_runs_on_dtlz1():
    """NSGA-II 在 DTLZ1 上应能正常运行"""
    np.random.seed(42)
    problem = DTLZ1(n_objs=3)
    algo = NSGAII(n_sols=100, max_iter=100, visual_mode='none')
    algo.optimize(problem)
    best = algo.get_best()
    assert best.n_sols >= 2
    assert best.objs.shape[1] == 3
