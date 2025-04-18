"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
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


class SOP5(PROBLEM):
    def __init__(self, num_dec=30, lower=-30, upper=30):
        """
        SOP5: Generalized Rosenbrock's Function

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param num_dec: 决策变量个数
        :param lower: 决策变量下界
        :param upper: 决策变量上界
        """
        num_obj = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        objs = np.sum(100 * (X[:, 1:] - X[:, :self.num_dec - 1] ** 2) ** 2 + (X[:, :self.num_dec - 1] - 1) ** 2, axis=1)
        return objs
