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


class MOP2(PROBLEM):
    def __init__(self, num_dec=3, lower=-4, upper=4):
        """
        MOP2

        References: Multi-objective evolutionary algorithm test suites,
        DA Van Veldhuizen, GB Lamont
        :param num_dec: 决策变量个数
        :param lower: 决策变量下界
        :param upper: 决策变量上界
        """
        num_obj = 2
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        f1 = 1 - np.exp(-np.sum((X - 1 / np.sqrt(self.num_dec)) ** 2, axis=1))
        f2 = 1 - np.exp(-np.sum((X + 1 / np.sqrt(self.num_dec)) ** 2, axis=1))
        objs = np.column_stack((f1, f2))
        return objs
