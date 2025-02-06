import numpy as np
from Algorithms.ALGORITHM import ALGORITHM


class FI(ALGORITHM):
    def __init__(self, problem):
        """
        最远插入启发式算法(Farthest Insertion)
        *Code Author: Luchen Wang
        :param problem: 问题对象(TSP类型)
        :param show_mode: 绘图模式
        """
        # 获取问题的距离矩阵
        self.dist_mat = problem.dist_mat
        super().__init__(problem, 1, len(self.dist_mat), None, None, show_mode=0)
        self.init_algorithm()

    @ALGORITHM.record_time
    def run(self):
        num_points = len(self.dist_mat)
        mask = np.zeros(num_points, dtype=bool)
        tour = []
        for i in self.iterator:
            # 得到候选下标
            cand_index = np.flatnonzero(mask == 0)
            if i == 0:
                # 找到最远的点
                chosen = self.dist_mat.max(axis=1).argmax()
                # 选择该点插入路由
                tour = [chosen]
                # 已选的点进行mask
                mask[chosen] = True
            else:
                # 从距离已选点最近的候选点中选择距离最远的插入
                chosen = cand_index[self.dist_mat[np.ix_(~mask, mask)].min(axis=1).argmax()]
                # 计算插入成本
                insert_cost = self.dist_mat[tour, chosen] + self.dist_mat[chosen, np.roll(tour, -1)] - self.dist_mat[
                    tour, np.roll(tour, -1)]
                # 计算插入位置
                insert_index = np.argmin(insert_cost)
                # 在路由中该位置插入节点
                tour.insert(insert_index + 1, chosen)
                # 已选的点进行mask
                mask[chosen] = True
        self.pop[0] = np.array(tour, dtype=int)
        self.objs = self.cal_objs(self.pop)
        self.record()
