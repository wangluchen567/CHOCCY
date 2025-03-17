import numpy as np
from Problems.PROBLEM import PROBLEM


class ZDT5(PROBLEM):
    def __init__(self, num_dec=None):
        problem_type = PROBLEM.BIN
        num_obj = 2
        if num_dec is None:
            num_dec = 80
        self.num_dec = np.ceil(max(num_dec - 30, 1) / 5) * 5 + 30
        lower = 0
        upper = 1
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        X = np.array(X)
        # 计算u向量
        u_temp = np.zeros((X.shape[0], 1 + (X.shape[1] - 30) // 5))
        # 第一列是前30列的和
        u_temp[:, 0] = np.sum(X[:, :30], axis=1)
        # 计算剩余列的u
        for i in range(1, u_temp.shape[1]):
            start_col = (i - 1) * 5 + 30
            end_col = start_col + 5
            u_temp[:, i] = np.sum(X[:, start_col:end_col], axis=1)
        # 计算 v向量
        v_temp = np.zeros_like(u_temp)
        v_temp[u_temp < 5] = 2 + u_temp[u_temp < 5]
        v_temp[u_temp == 5] = 1
        # 初始化目标值
        objs = np.zeros((len(X), 2))
        objs[:, 0] = 1 + u_temp[:, 0]  # 第一列目标值
        g = np.sum(v_temp[:, 1:], axis=1)  # g 是 v 的第二列到最后一列的和
        h = 1 / objs[:, 0]  # h 是第一列目标值的倒数
        objs[:, 1] = g * h  # 第二列目标值
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = np.zeros((31, 2))
        optimums[:, 0] = np.arange(1, 32)
        optimums[:, 1] = (self.num_dec - 30) / 5 / optimums[:, 0]
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿"""
        return self.get_optimum(N)
