"""
聚合函数测试（验证 PBI d2 修正的正确性）
"""
import numpy as np
from choccy.utilities.commons import aggregate


def test_pbi_single():
    """PBI 单个体模式应正确运行"""
    objs = np.array([[1.0, 2.0]])
    weights = np.array([[0.5, 0.5]])
    ref_point = np.array([0.0, 0.0])
    result = aggregate(objs, weights, ref_point, method='pbi')
    assert result.shape == (1,)
    assert result[0] > 0


def test_pbi_multi():
    """PBI 多个体模式应正确运行"""
    objs = np.array([[1.0, 2.0], [3.0, 1.0]])
    weights = np.array([[0.5, 0.5], [0.3, 0.7]])
    ref_point = np.array([0.0, 0.0])
    result = aggregate(objs, weights, ref_point, method='pbi')
    assert result.shape == (2,)
    assert np.all(result > 0)


def test_tchebycheff():
    """切比雪夫聚合"""
    objs = np.array([[1.0, 2.0]])
    weights = np.array([[0.5, 0.5]])
    ref_point = np.array([0.0, 0.0])
    result = aggregate(objs, weights, ref_point, method='tch')
    assert result.shape == (1,)
    assert result[0] > 0


def test_weighted_sum():
    """加权和聚合"""
    objs = np.array([[1.0, 2.0]])
    weights = np.array([[0.5, 0.5]])
    ref_point = np.array([0.0, 0.0])
    result = aggregate(objs, weights, ref_point, method='wsm')
    assert result.shape == (1,)
    assert abs(result[0] - 1.5) < 1e-10


def test_pbi_d2_correct_with_unit_weight():
    """验证 d2 公式修正后的正确性（与勾股定理等价）"""
    objs = np.array([[1.0, 2.0, 3.0]])
    weights = np.array([[0.2, 0.3, 0.5]])
    ref_point = np.array([0.0, 0.0, 0.0])
    # 使用 PBI 计算
    result = aggregate(objs, weights, ref_point, method='pbi', theta=5.0)
    # 手动验证：用勾股定理计算等价结果
    norm_w = np.linalg.norm(weights)
    proj_len = np.dot(objs[0], weights[0]) / norm_w
    norm_v = np.linalg.norm(objs[0])
    d2_pythag = np.sqrt(max(0, norm_v**2 - proj_len**2))
    expected = proj_len + 5.0 * d2_pythag
    assert abs(result[0] - expected) < 1e-10
