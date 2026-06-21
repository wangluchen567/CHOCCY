"""
多属性决策筛选方法测试
"""
import numpy as np
from choccy.utilities.commons import select_by_topsis, select_by_cosine, select_by_vikor


class TestTopsis:
    def test_selects_balanced(self):
        objs = np.array([[0.2, 0.9], [0.5, 0.5], [0.8, 0.2]])
        idx = select_by_topsis(objs)
        assert idx == 1

    def test_with_weights(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_topsis(objs, weights=[0.9, 0.1])
        assert idx == 0

    def test_reverse_weight(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_topsis(objs, weights=[0.1, 0.9])
        assert idx == 1

    def test_single_solution(self):
        idx = select_by_topsis(np.array([[0.5, 0.5]]))
        assert idx == 0


class TestCosine:
    def test_selects_by_weight(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_cosine(objs, [0.9, 0.1], 2)
        assert idx == 1  # (0.5,0.5)方向更接近

    def test_reverse_weight(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_cosine(objs, [0.1, 0.9], 2)
        assert idx == 0  # (0.1,0.9)方向更接近

    def test_zero_weight_fallback(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_cosine(objs, [0, 0], 2)
        assert idx == 1


class TestVikor:
    def test_selects_balanced(self):
        objs = np.array([[0.2, 0.9], [0.5, 0.5], [0.8, 0.2]])
        idx = select_by_vikor(objs)
        assert idx == 1

    def test_with_weights(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_vikor(objs, weights=[0.9, 0.1])
        assert idx == 0

    def test_reverse_weight(self):
        objs = np.array([[0.1, 0.9], [0.5, 0.5]])
        idx = select_by_vikor(objs, weights=[0.1, 0.9])
        assert idx == 1

    def test_single_solution(self):
        idx = select_by_vikor(np.array([[0.5, 0.5]]))
        assert idx == 0
