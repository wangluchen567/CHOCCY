import numpy as np
import matplotlib.pyplot as plt
from Problems.PROBLEM import PROBLEM
from mpl_toolkits.mplot3d import Axes3D


class FixLabelCluster(PROBLEM):
    def __init__(self, num_dec=120, lower=1, upper=4):
        problem_type = PROBLEM.FIX
        num_obj = 1
        self.example_dec = np.repeat(np.arange(lower, upper), int(num_dec / (upper - lower)))
        super().__init__(problem_type, num_dec, num_obj, lower, upper)
        # 随机生成数据
        self.points = np.random.uniform(0, 1, size=(self.num_dec, 2))

    def _cal_objs(self, X):
        num_sol = len(X)
        objs = np.zeros(num_sol)
        types = np.unique(X[0])
        points = np.repeat(self.points[np.newaxis, :, :], repeats=num_sol, axis=0)
        for t in types:
            # 得到该类中的所有点
            this_type_points = points[np.where(X == t)].reshape(num_sol, -1, self.points.shape[-1])
            # 得到该类的中心点坐标
            centroids = np.mean(this_type_points, axis=1)
            # 计算与中心点之间的距离
            distances = np.linalg.norm(this_type_points - centroids[:, np.newaxis, :], axis=-1)
            objs += np.sum(distances, axis=1)
        return objs

    def plot_(self, best, pause=False, n_iter=None, pause_time=0.1):
        if not pause: plt.figure()
        plt.clf()
        plt.scatter(self.points[:, 0], self.points[:, 1], c=best, cmap='rainbow')
        plt.grid()
        if pause:
            if n_iter is not None:
                plt.title("iter: " + str(n_iter))
            plt.pause(pause_time)
        else:
            plt.show()
