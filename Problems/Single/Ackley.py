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


class Ackley(PROBLEM):
    def __init__(self, num_dec=30):
        num_obj = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower=-32, upper=32)

    def _cal_objs(self, X):
        objs = -20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, axis=1) / self.num_dec)) - np.exp(
            np.sum(np.cos(2 * np.pi * X), axis=1) / self.num_dec) + 20 + np.e
        return objs
