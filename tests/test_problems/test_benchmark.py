"""
基准问题测试：验证问题定义的正确性
"""
import numpy as np
from choccy.problems.single import Sphere, Ackley
from choccy.problems.multi import ZDT1, DTLZ1


def test_sphere_optimum():
    """Sphere 函数在原点应为 0"""
    problem = Sphere(n_vars=5)
    xs = np.zeros((1, 5))
    objs = problem.calc_objs(xs)
    assert abs(objs[0, 0]) < 1e-10


def test_ackley_optimum():
    """Ackley 函数在原点应为 0"""
    problem = Ackley(n_vars=5)
    xs = np.zeros((1, 5))
    objs = problem.calc_objs(xs)
    assert abs(objs[0, 0]) < 1e-10


def test_zdt1_pareto_front():
    """ZDT1 的 Pareto 前沿性质：f2 = 1 - sqrt(f1)"""
    problem = ZDT1()
    optimums = problem.get_optimums()
    f1 = optimums[:, 0]
    f2 = optimums[:, 1]
    expected = 1 - np.sqrt(f1)
    assert np.allclose(f2, expected, atol=1e-10)


def test_dtlz1_pareto_front():
    """DTLZ1 的 Pareto 前沿性质：sum(f) = 0.5"""
    problem = DTLZ1(n_objs=3)
    optimums = problem.get_optimums()
    assert np.allclose(np.sum(optimums, axis=1), 0.5, atol=1e-10)
