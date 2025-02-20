import numpy as np
import matplotlib.pyplot as plt
from Problems.PROBLEM import PROBLEM
from mpl_toolkits.mplot3d import Axes3D


class MixFixLabelCluster(PROBLEM):
    """混合固定标签聚类"""
    def __init__(self, num_points=120, lower=1, upper=4, real_lower=0, real_upper=1):
        num_obj = 1
        self.num_points = num_points
        num_dec = self.num_points + (upper - lower) * 2
        # 决策变量由两部分组成，第一部分为固定标签类型，第二部分为实数类型
        problem_type = np.zeros(num_dec, dtype=int)
        problem_type[:self.num_points] = PROBLEM.FIX
        # 给定固定标签类型的示例
        self.example_dec = np.repeat(np.arange(lower, upper), int(self.num_points / (upper - lower)))
        # 重新指定上下界
        lower = np.zeros(num_dec) + lower
        upper = np.zeros(num_dec) + upper
        lower[self.num_points:] = real_lower
        upper[self.num_points:] = real_upper
        super().__init__(problem_type, num_dec, num_obj, lower, upper)
        # 随机生成数据
        self.points = np.random.uniform(real_lower, real_upper, size=(self.num_points, 2))
        self.weights = np.random.uniform(real_lower, real_upper, size=self.num_points)

    def _cal_objs(self, X):
        num_sol = len(X)
        objs = np.zeros(num_sol)
        types = np.unique(X[0, :self.num_points])
        all_centroids = X[:, self.num_points:].reshape(num_sol, -1, 2)
        points = np.repeat(self.points[np.newaxis, :, :], repeats=num_sol, axis=0)
        weights = np.repeat(self.weights[np.newaxis, :], repeats=num_sol, axis=0)
        for i in range(len(types)):
            t = types[i]
            # 得到该类中的所有点
            this_type_points = points[np.where(X[:, :self.num_points] == t)].reshape(num_sol, -1, self.points.shape[-1])
            this_type_weights = weights[np.where(X[:, :self.num_points] == t)].reshape(num_sol, -1)
            # 得到该类的中心点坐标
            centroids = all_centroids[:, i, :]
            # 计算与中心点之间的距离
            distances = np.linalg.norm(this_type_points - centroids[:, np.newaxis, :], axis=-1)
            distances_weights = distances * this_type_weights
            objs += np.sum(distances_weights, axis=1)
        return objs

    def plot(self, best, pause=False, n_iter=None, pause_time=0.1):
        if not pause: plt.figure()
        plt.clf()
        best_types = best[:self.num_points]
        best_centroids = best[self.num_points:].reshape(-1, 2)
        plt.scatter(self.points[:, 0], self.points[:, 1], c=best_types, cmap='rainbow')
        plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='black', marker='x')
        plt.grid()
        if pause:
            if n_iter is not None:
                plt.title("iter: " + str(n_iter))
            plt.pause(pause_time)
        else:
            plt.show()
