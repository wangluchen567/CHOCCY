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


class DTLZ5(PROBLEM):
    def __init__(self, num_dec=None, num_obj=None, lower=0, upper=1):
        problem_type = PROBLEM.REAL
        if num_obj is None:
            num_obj = 3
        if num_dec is None:
            num_dec = num_obj + 9
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        M = self.num_obj
        g = np.sum((X[:, M - 1:] - 0.5) ** 2, axis=1)
        Temp = np.tile(g, (M - 2, 1)).T
        X[:, 1:M - 1] = (1 + 2 * Temp * X[:, 1:M - 1]) / (2 + 2 * Temp)
        objs = (np.tile(1 + g, (M, 1)).T *
                  np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1), dtype=float),
                                                  np.cos(X[:, :M - 1] * np.pi / 2))), axis=1)) *
                  np.hstack((np.ones((g.shape[0], 1), dtype=float),
                             np.sin(X[:, M - 2::-1] * np.pi / 2))))

        return objs
