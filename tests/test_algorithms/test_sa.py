"""
SA 算法测试
"""
import numpy as np
from choccy.algorithms.single import SA
from choccy.problems.single import Sphere, Ackley


def test_sa_converges_on_sphere():
    """SA 在低维 Sphere 上应能收敛"""
    np.random.seed(42)
    problem = Sphere(n_vars=5)
    algo = SA(max_iter=5000, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 0.1


def test_sa_runs_with_constraints():
    """SA 使用惩罚系数应能正常运行"""
    np.random.seed(42)
    problem = Sphere(n_vars=5)
    algo = SA(max_iter=1000, penalty_coef=1e6, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best() is not None


def test_sa_converges_with_high_iterations():
    """增加迭代次数后 SA 应收敛更好"""
    np.random.seed(42)
    problem = Sphere(n_vars=3)
    algo = SA(max_iter=10000, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 0.05
