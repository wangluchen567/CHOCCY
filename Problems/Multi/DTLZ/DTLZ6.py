"""
Copyright (c) 2024 LuChen Wang
[Software Name] is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
from Problems import PROBLEM


class DTLZ6(PROBLEM):
    def __init__(self, num_dec=None, num_obj=None, lower=0, upper=1):
        problem_type = PROBLEM.REAL
        if num_obj is None:
            num_obj = 3
        if num_dec is None:
            num_dec = num_obj + 9
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X_):
        X = X_.copy()
        M = self.num_obj
        g = np.sum(np.clip(X[:, M - 1:], a_min=0, a_max=None) ** 0.1, axis=1)
        Temp = np.tile(g[:, np.newaxis], (1, M - 2))
        X[:, 1:M - 1] = (1 + 2 * Temp * X[:, 1:M - 1]) / (2 + 2 * Temp)
        objs = np.tile(1 + g[:, np.newaxis], (1, M))
        cumprod_part = np.fliplr(
            np.cumprod(np.hstack((np.ones((g.shape[0], 1)), np.cos(X[:, :M - 1] * np.pi / 2))), axis=1))
        sin_part = np.hstack((np.ones((g.shape[0], 1)), np.sin(X[:, M - 2::-1] * np.pi / 2)))
        objs = objs * cumprod_part * sin_part
        return objs