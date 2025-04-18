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


class SOP9(PROBLEM):
    def __init__(self, num_dec=30):
        """
        SOP9: Generalized Rastrigin's Function

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param num_dec: 决策变量个数
        """
        num_obj = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower=-5.12, upper=5.12)

    def _cal_objs(self, X):
        objs = np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)
        return objs
