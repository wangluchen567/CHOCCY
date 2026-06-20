"""
PSO 算法测试
"""
import numpy as np
from choccy.algorithms.single import PSO
from choccy.problems.single import Sphere, Ackley


def test_pso_converges_on_sphere():
    """PSO 应在 200 代内收敛到接近 0"""
    np.random.seed(42)
    problem = Sphere(n_vars=10)
    algo = PSO(n_sols=50, max_iter=200, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 1e-5


def test_pso_runs_with_dynamic_weight():
    """惯性权重衰减模式能正常运行"""
    np.random.seed(42)
    problem = Ackley(n_vars=10)
    algo = PSO(n_sols=50, max_iter=1000, w=(0.9, 0.3), c1=2.0, c2=2.0, visual_mode='none')
    algo.optimize(problem)
    assert algo.get_best().f < 1.e-5


def test_pso_boundary_handling():
    """所有历史位置应在边界内"""
    np.random.seed(42)
    problem = Ackley(n_vars=5)
    algo = PSO(visual_mode='none')
    algo.optimize(problem)
    for xs in algo.history_xs:
        assert np.all(xs >= problem.l_bounds - 1e-10)
        assert np.all(xs <= problem.u_bounds + 1e-10)
