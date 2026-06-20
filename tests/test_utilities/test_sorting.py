"""
非支配排序和拥挤度测试
"""
import numpy as np
from choccy.utilities.commons import fast_nd_sort, crowding_dist, composite_rank, is_dom


def test_is_dom():
    """支配关系测试"""
    p = np.array([1, 2])
    q = np.array([3, 4])
    assert is_dom(p, q) is True   # p 支配 q
    assert is_dom(q, p) is False  # q 不支配 p


def test_is_dom_non_dominated():
    """相互非支配测试"""
    p = np.array([1, 4])
    q = np.array([3, 2])
    assert is_dom(p, q) is False
    assert is_dom(q, p) is False


def test_fast_nd_sort():
    """非支配排序测试：3 个解，2 个在前沿"""
    objs = np.array([
        [1, 2],   # 前沿
        [2, 1],   # 前沿
        [3, 3],   # 被支配
    ])
    fronts, ranks = fast_nd_sort(objs)
    assert len(fronts) >= 1
    assert 0 in fronts[0] and 1 in fronts[0]  # (1,2) 和 (2,1) 在前沿


def test_crowding_dist():
    """拥挤度距离：边界点应为 inf"""
    objs = np.array([
        [0, 1],
        [0.5, 0.5],
        [1, 0],
    ])
    fronts = [[0, 1, 2]]
    dist = crowding_dist(objs, fronts)
    assert np.isinf(dist[0])   # 边界点
    assert np.isinf(dist[2])   # 边界点
    assert 0 <= dist[1] < np.inf  # 中间点


def test_composite_rank():
    """综合排名：rank 优先，同 rank 下拥挤度大的靠前"""
    ranks = np.array([1, 1, 2])
    crowd = np.array([1.0, 2.0, 1.0])
    ranking = composite_rank(ranks, crowd)
    # rank=1 的应在 rank=2 的前面
    assert ranking[2] > ranking[0] and ranking[2] > ranking[1]
