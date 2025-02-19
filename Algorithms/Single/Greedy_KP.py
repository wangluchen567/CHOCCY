import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM


class Greedy_KP(ALGORITHM):
    def __init__(self, problem):
        super().__init__(problem, num_pop=None, num_iter=None, cross_prob=None, mutate_prob=None, show_mode=0)
        # 问题必须为单目标问题
        if problem.num_obj > 1:
            raise ValueError("This method can only solve single objective problems")
        # 问题必须为二进制问题
        if np.sum(problem.problem_type != ALGORITHM.BIN):
            raise ValueError("This method can only solve binary problems")
        # 问题必须为背包问题
        if hasattr(problem, 'weights') and hasattr(problem, 'values') and hasattr(problem, 'capacity'):
            self.weighs = problem.weights
            self.values = problem.values
            self.capacity = problem.capacity
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        cost = (self.values / self.weighs).flatten()
        cost_sort = np.argsort(-cost)
        Sum_Weight = 0
        chosen = []
        for i in tqdm(range(len(cost_sort))):
            if Sum_Weight == self.capacity:
                break
            if Sum_Weight > self.capacity:
                last = chosen.pop()
                Sum_Weight -= self.weighs[last]
                break
            chosen.append(cost_sort[i])
            Sum_Weight += self.weighs[cost_sort[i]]
        solution = np.zeros(len(self.weighs), dtype=int)
        solution[chosen] = 1
        self.best = solution
        self.best_obj = self.cal_objs(self.best)[0]
        self.best_con = self.cal_cons(self.best)[0]
