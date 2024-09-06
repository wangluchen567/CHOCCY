import numpy as np


class PROBLEM():
    def __init__(self, problem_type, num_dec, num_obj, lower, upper):
        self.problem_type = problem_type
        self.num_dec = num_dec
        self.num_obj = num_obj
        self.lower = lower
        self.upper = upper
        self.optimums = self.get_optimum()
        self.pareto_front = self.get_pareto_front()
        self.format_range()

    def format_range(self):
        if isinstance(self.problem_type, int):
            self.problem_type = np.zeros(self.num_dec) + self.problem_type
        if isinstance(self.lower, int) or isinstance(self.lower, float):
            self.lower = np.zeros(self.num_dec) + self.lower
        if isinstance(self.upper, int) or isinstance(self.upper, float):
            self.upper = np.zeros(self.num_dec) + self.upper

    def cal_objs(self, X):
        raise NotImplemented

    def cal_cons(self, X):
        return -np.ones(len(X))

    def get_optimum(self, *args, **kwargs):
        pass

    def get_pareto_front(self, *args, **kwargs):
        pass

    def plot_(self, *args, **kwargs):
        pass
