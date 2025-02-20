import warnings
import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM


class DP_KP(ALGORITHM):
    def __init__(self, problem):
        super().__init__(problem, num_pop=1)
        # 问题必须为单目标问题
        if problem.num_obj > 1:
            raise ValueError("This method can only solve single objective problems")
        # 问题必须为二进制问题
        if np.sum(problem.problem_type != ALGORITHM.BIN):
            raise ValueError("This method can only solve binary problems")
        # 问题必须为背包问题
        if hasattr(problem, 'weights') and hasattr(problem, 'values') and hasattr(problem, 'capacity'):
            self.weighs = problem.weights.astype(int).flatten()
            self.values = problem.values.astype(int).flatten()
            self.capacity = int(problem.capacity)
            # 问题中的数据必须是整数
            if not (problem.weights.dtype == np.int and
                    problem.values.dtype == np.int and
                    isinstance(problem.capacity, int)):
                warnings.warn(
                    "This method can only solve integer type knapsack problems, "
                    "and the provided dataset will be forcibly converted to integer type")
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        N = len(self.weighs)
        dp = [0] * (self.capacity + 1)
        # selected 用于记录在每个dp状态下选择的物品
        selected = [[0] * N for _ in range(self.capacity + 1)]
        for i in tqdm(range(N)):
            for j in range(self.capacity, self.weighs[i] - 1, -1):
                if dp[j] < dp[j - self.weighs[i]] + self.values[i]:
                    dp[j] = dp[j - self.weighs[i]] + self.values[i]
                    # 更新选择状态
                    selected[j] = selected[j - self.weighs[i]].copy()
                    selected[j][i] = 1
        self.pop = np.array([selected[self.capacity]])
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        self.record()

    def get_best(self, **kwargs):
        return self.pop[0], self.objs[0], self.cons[0]
