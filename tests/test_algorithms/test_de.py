"""
DE 算法测试
"""
import numpy as np
from choccy.algorithms.single import DE
from choccy.problems.single import Sphere


def test_de_rand1_converges():
    """DE/rand/1 应在 Sphere 上收敛"""
    np.random.seed(42)
    problem = Sphere(n_vars=10)
    algo = DE(n_sols=50, max_iter=200, operator_type=DE.RAND_1, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 1e-5


def test_de_best1_converges():
    """DE/best/1 应能运行且收敛"""
    np.random.seed(42)
    problem = Sphere(n_vars=10)
    algo = DE(n_sols=50, max_iter=200, operator_type=DE.BEST_1, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 1e-5


def test_de_strict_indices():
    """严格索引模式也能正常收敛"""
    np.random.seed(42)
    problem = Sphere(n_vars=5)
    algo = DE(quick_indices=False, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 0.1
