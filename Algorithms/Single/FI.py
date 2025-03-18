import numpy as np
from Algorithms.ALGORITHM import ALGORITHM


class FI(ALGORITHM):
    def __init__(self, show_mode=0):
        """
        最远插入启发式算法(Farthest Insertion)
        *Code Author: Luchen Wang
        :param show_mode: 绘图模式
        """
        super().__init__(num_pop=1, num_iter=None, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.PMU]
        self.dist_mat = None

    @ALGORITHM.record_time
    def init_algorithm(self, problem, pop=None):
        super().init_algorithm(problem, pop)
        # 问题必须提供距离矩阵
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
        # 初始化迭代次数
        self.num_iter = self.num_dec
        # 获取问题的距离矩阵
        self.dist_mat = self.problem.dist_mat  # type: ignore

    @ALGORITHM.record_time
    def run(self):
        num_points = len(self.dist_mat)
        mask = np.zeros(num_points, dtype=bool)
        tour = []
        for i in self.get_iterator():
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
        self.pop = np.array([tour], dtype=int)
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        # 清空所有记录后重新记录
        self.clear_record()
        self.record()

    def get_current_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
