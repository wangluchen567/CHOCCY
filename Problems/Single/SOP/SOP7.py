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


class SOP7(PROBLEM):
    def __init__(self, num_dec=30):
        """
        SOP7: Quartic Function with Noise

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param num_dec: 决策变量个数
        """
        num_obj = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower=-1.28, upper=1.28)

    def _cal_objs(self, X):
        objs = (np.sum(np.arange(1, self.num_dec + 1)[np.newaxis, :].repeat(len(X), axis=0) * X ** 4, axis=1) +
                np.random.rand(X.shape[0]))
        return objs
