import os
import numpy as np
from Problems.PROBLEM import PROBLEM


class KP(PROBLEM):
    def __init__(self, num_dec=100, weights=None, values=None, capacity=None):
        problem_type = 1
        num_obj = 1
        lower = 0
        upper = 1
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

        if (weights is not None) and (values is not None) and (capacity is not None):
            # 若给定参数均非空，则根据指定参数
            self.weights = weights
            self.values = values
            self.capacity = capacity
        elif (weights is None) and (values is None) and (capacity is None):
            # 若给定参数均为空，则需要检查当前路径是否有数据集
            # 得到项目的根目录
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2))
            # 保存到Datasets中
            file_name = project_root + "\\Datasets\\Single\\KP-" + str(self.num_dec) + ".txt"
            if os.path.isfile(file_name):
                data = np.loadtxt(file_name, delimiter=',')
                self.weights = data[:, 0]
                self.values = data[:, 1]
                self.capacity = data[0, 2]
            else:
                # 若没有数据集则随机生成数据集并保存
                self.weights = np.random.randint(10, 100, size=self.num_dec)
                self.values = np.random.randint(10, 100, size=self.num_dec)
                self.capacity = int(np.sum(self.weights) / 2)
                # self.weights = np.random.uniform(10, 100, size=num_dec)
                # self.values = np.random.uniform(10, 100, size=num_dec)
                # self.capacity = np.sum(self.weights) / 2
                # 保存数据集
                data = np.zeros(shape=(self.num_dec, 3), dtype=int)
                data[:, 0], data[:, 1] = self.weights, self.values
                data[0, 2] = self.capacity
                np.savetxt(file_name, data, fmt="%d", delimiter=',')
        else:
            raise ValueError("All three parameters (weights, values, capacity) must be provided, "
                             "not just a portion")
        # 若给定的数据集不是纵向排布的则进行转换
        if self.weights.ndim == 1:
            self.weights = self.weights.reshape(-1, 1)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        # 储存实例数据集以便 NNDREA 使用
        self.instance = np.hstack((self.weights, self.values))

    def cal_objs(self, X):
        objs = np.sum(self.values) - X.dot(self.values)
        cons = self.cal_cons(X)
        feas = cons <= 0
        objs = feas * objs + (1 - feas) * (cons + 1e10)
        return objs

    def cal_cons(self, X):
        cons = X.dot(self.weights) - self.capacity
        return cons
