import numpy as np
from Algorithms.ALGORITHM import ALGORITHM


class GreedyKP(ALGORITHM):
    def __init__(self, problem, show_mode=0):
        super().__init__(problem, num_pop=1, num_iter=problem.num_dec, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.BIN]
        self.weights = None
        self.values = None
        self.capacity = None

    @ALGORITHM.record_time
    def init_algorithm(self, pop=None):
        super().init_algorithm(pop)
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
        for i in self.iterator:
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
        self.pop = np.array([solution])
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        # 清空所有记录后重新记录
        self.clear_record()
        self.record()

    def get_current_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
