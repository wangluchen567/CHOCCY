import numpy as np
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Utils import get_uniform_vectors


class MOEAD(ALGORITHM):
    def __init__(self, problem, num_pop=100, num_iter=100, func_type=1,
                 cross_prob=None, mutate_prob=None, show_mode=0):
        """
        This code is based on the research presented in
        "MOEA/D: A multi-objective evolutionary algorithm based on decomposition"
        by Q. Zhang and H. Li
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param func_type: 聚合函数类型
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        # 聚合函数类型
        self.func_type = func_type
        # 选择的最近邻居的数量
        self.num_near = int(np.ceil(num_pop / 10))
        # 均匀生成权重向量
        self.vectors = get_uniform_vectors(num_pop, problem.num_obj)
        # 获取每个权重向量的前T个邻居向量的下标
        self.indexes = self.get_neighbor_index(self.vectors, self.num_near)
        # 根据权重向量个数重新确定种群大小(必须匹配)
        num_pop = len(self.vectors)
        # 初始化相关参数(调用父类初始化)
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, show_mode)
        self.init_algorithm()
        # 初始化参考点
        self.ref = np.min(self.objs, axis=0)

    @ALGORITHM.record_time
    def run(self):
        """运行算法(主函数)"""
        # 绘制初始状态图
        self.plot(pause=True, n_iter=0)
        for i in self.iterator:
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)

    def run_step(self, i):
        """运行算法单步"""
        for j in range(self.num_pop):
            # 随机选择两个个体作为父代个体
            mating_pool = self.selection_single(j)
            # 进行交叉变异得到一个新的子代
            offspring = self.operator(mating_pool)[0]
            # 进行环境选择更新种群
            self.environmental_selection_single(offspring, j)
        # 记录每步状态
        self.record(i + 1)

    def selection_single(self, j):
        return np.random.choice(self.indexes[j], size=2, replace=False)

    def environmental_selection_single(self, offspring, j):
        """进行环境选择(分解式)"""
        # 计算该子代的目标值
        offspring_obj = self.cal_objs(offspring)
        offspring_con = self.cal_cons(offspring)
        # 更新参考点
        self.ref = np.min((offspring_obj.flatten(), self.ref), axis=0)
        # 对新解的所有邻居解进行更新
        neighbors = self.indexes[j]
        # 使用指定聚合函数计算后选择更优的个体
        better = self.aggregate(self.vectors[neighbors], offspring_obj) <= self.aggregate(self.vectors[neighbors],
                                                                                          self.objs[neighbors])
        # 更新种群和目标值
        self.pop[neighbors[better]] = offspring
        self.objs[neighbors[better]] = offspring_obj
        self.cons[neighbors[better]] = offspring_con

    @staticmethod
    def get_neighbor_index(weights, t):
        """
        获取每个权重向量的前T个邻居向量的下标
        :param weights: 权重向量
        :param t: 最近邻居的数量
        :return: 前t个邻居向量的下标
        """
        # 计算欧式距离矩阵
        dist_matrix = np.sqrt(np.sum((weights[:, np.newaxis, :] - weights[np.newaxis, :, :]) ** 2, axis=2))
        # 获取前T个最近的邻居的下标
        return np.argsort(dist_matrix, axis=1)[:, :t]

    def aggregate(self, vectors, objs):
        # 改变形状方便矩阵运算
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
        if self.func_type == 1:
            # 基于惩罚边界的聚合方法
            theta = 5  # 设置超参数
            if len(objs) == 1:
                # 若是单个个体则直接求
                d1 = np.abs(np.dot(objs - self.ref, vectors.T)).flatten() / np.linalg.norm(vectors, axis=1)
                d2 = np.linalg.norm(objs - (self.ref + d1.reshape(-1, 1) * vectors), axis=1)
            else:
                # 若是需要对整个邻居目标值则使用取对角线方法
                d1 = np.abs(np.diag(np.dot(objs - self.ref, vectors.T))) / np.linalg.norm(vectors, axis=1)
                d2 = np.linalg.norm(objs - (self.ref + d1.reshape(-1, 1) * vectors), axis=1)
            return d1 + theta * d2
        elif self.func_type == 2:
            # 切比雪夫聚合方法
            return np.max(vectors * np.abs(objs - self.ref), axis=1)
        elif self.func_type == 3:
            # 线性聚合方法
            if len(objs) == 1:
                # 若是单个个体则直接求点积
                return np.dot(objs, vectors.T).flatten()
            else:
                # 若是需要对整个邻居目标值则使用 取对角线方法
                return np.diag(np.dot(objs, vectors.T)).flatten()
                # 使用爱因斯坦求和约定对矩阵逐行求点积
                # return np.einsum('ij,ij->i', objs, vectors).flatten()
        else:
            raise ValueError("There is no such aggregate function type")
