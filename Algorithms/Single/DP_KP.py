import warnings
import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM


class DPKP(ALGORITHM):
    def __init__(self, problem, show_mode=0):
        """
        动态规划求解背包问题(KP)
        *Code Author: Luchen Wang
        :param problem: 问题对象(必须是背包问题)
        :param
        """
        super().__init__(problem, num_pop=1, show_mode=show_mode)
        self.weights = None
        self.values = None
        self.capacity = None

    def init_algorithm(self):
        # 问题必须为单目标问题
        if self.problem.num_obj > 1:
            raise ValueError("This method can only solve single objective problems")
        # 问题必须为二进制问题
        if np.sum(self.problem.problem_type != ALGORITHM.BIN):
            raise ValueError("This method can only solve binary problems")
        # 问题必须为背包问题
        if hasattr(self.problem, 'weights') and hasattr(self.problem, 'values') and hasattr(self.problem, 'capacity'):
            self.weights = self.problem.weights.astype(int).flatten()
            self.values = self.problem.values.astype(int).flatten()
            self.capacity = int(self.problem.capacity)
            # 问题中的数据必须是整数
            if not ('int' in self.problem.weights.dtype.name and
                    'int' in self.problem.values.dtype.name and
                    isinstance(self.problem.capacity, int)):
                warnings.warn(
                    "This method can only solve integer type knapsack problems, "
                    "and the provided dataset will be forcibly converted to integer type")
        else:
            raise ValueError("This method can only solve knapsack problems")
        # 构建迭代器
        self.iterator = tqdm(range(self.problem.num_dec)) if self.show_mode == 0 else range(self.problem.num_dec)

    @ALGORITHM.record_time
    def run(self):
        # 初始化算法
        self.init_algorithm()
        num_items = len(self.weights)
        dp = [0] * (self.capacity + 1)
        # selected 用于记录在每个dp状态下选择的物品
        selected = [[0] * num_items for _ in range(self.capacity + 1)]
        for i in self.iterator:
            for j in range(self.capacity, self.weights[i] - 1, -1):
                if dp[j] < dp[j - self.weights[i]] + self.values[i]:
                    dp[j] = dp[j - self.weights[i]] + self.values[i]
                    # 更新选择状态
                    selected[j] = selected[j - self.weights[i]].copy()
                    selected[j][i] = 1
        self.pop = np.array([selected[self.capacity]])
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        self.record()

    def get_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
