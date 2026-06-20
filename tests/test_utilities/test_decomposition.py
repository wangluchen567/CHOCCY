"""
权重生成测试
"""
import numpy as np
from choccy.utilities.commons import generate_uniform_weights


def test_weights_sum_to_one():
    """所有权重向量应归一化到和为 1"""
    weights = generate_uniform_weights(100, 3)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-10)


def test_weights_positive():
    """所有权重分量应大于 0"""
    weights = generate_uniform_weights(100, 3)
    assert np.all(weights > 0)


def test_weights_count():
    """生成的权重数量应至少为请求数的一半（Das & Dennis 近似）"""
    weights = generate_uniform_weights(100, 3)
    # 3 目标下 H=12 生成 91 个权重，这是 Das & Dennis 的近似特性
    assert len(weights) >= 50
