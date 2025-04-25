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


class MOP1(PROBLEM):
    def __init__(self, lower=-1e3, upper=1e3):
        """
        MOP1

        References: Multi-objective evolutionary algorithm test suites,
        DA Van Veldhuizen, GB Lamont
        :param lower: 决策变量下界
        :param upper: 决策变量上界
        """
        num_dec = 1
        num_obj = 2
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        f1 = X ** 2
        f2 = (X - 2) ** 2
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = np.zeros((N, 2))
        optimums[:, 0] = np.linspace(0, 4, N)
        optimums[:, 1] = (np.sqrt(optimums[:, 0]) - 2) ** 2
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿(以绘图)"""
        return self.get_optimum(N)
