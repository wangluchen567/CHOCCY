import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM


class GreedyKP(ALGORITHM):
    def __init__(self, problem, show_mode=0):
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
            self.weights = self.problem.weights
            self.values = self.problem.values
            self.capacity = self.problem.capacity
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        self.init_algorithm()  # 初始化算法
        cost = (self.values / self.weights).flatten()
        cost_sort = np.argsort(-cost)
        sum_weight = 0
        chosen = []
        for i in tqdm(range(len(cost_sort))):
            if sum_weight == self.capacity:
                break
            if sum_weight > self.capacity:
                last = chosen.pop()
                sum_weight -= self.weights[last]
                break
            chosen.append(cost_sort[i])
            sum_weight += self.weights[cost_sort[i]]
        solution = np.zeros(len(self.weights), dtype=int)
        solution[chosen] = 1
        self.pop = np.repeat(np.array([solution]), len(self.pop), axis=0)
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        self.record()

    def get_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
