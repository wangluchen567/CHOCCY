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


class DTLZ7(PROBLEM):
    def __init__(self, num_dec=None, num_obj=None, lower=0, upper=1):
        problem_type = PROBLEM.REAL
        if num_obj is None:
            num_obj = 3
        if num_dec is None:
            num_dec = num_obj + 19
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        M = self.num_obj
        objs = np.zeros((X.shape[0], M))
        g = 1 + 9 * np.mean(X[:, M - 1:], axis=1)
        objs[:, :M - 1] = X[:, :M - 1]
        term1 = objs[:, :M - 1] / (1 + np.tile(g, (M - 1, 1)).T)
        term2 = 1 + np.sin(3 * np.pi * objs[:, :M - 1])
        objs[:, M - 1] = (1 + g) * (M - np.sum(term1 * term2, axis=1))
        return objs
