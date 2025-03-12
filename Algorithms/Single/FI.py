import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM


class FI(ALGORITHM):
    def __init__(self, problem, show_mode=0):
        """
        最远插入启发式算法(Farthest Insertion)
        *Code Author: Luchen Wang
        :param problem: 问题对象(TSP类型)
        """
        super().__init__(problem, num_pop=1, show_mode=show_mode)
        self.dist_mat = None

    @ALGORITHM.record_time
    def init_algorithm(self):
        # 问题必须提供距离矩阵
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
        # 获取问题的距离矩阵
        self.dist_mat = self.problem.dist_mat  # type: ignore
        # 问题必须为单目标问题
        if self.problem.num_obj > 1:
            raise ValueError("This method can only solve single objective problems")
        # 问题必须为序列问题
        if np.sum(self.problem_type != ALGORITHM.PMU):
            raise ValueError("This method can only solve sequence problems")
        # 构建迭代器
        self.iterator = tqdm(range(self.problem.num_dec)) if self.show_mode == 0 else range(self.problem.num_dec)

    @ALGORITHM.record_time
    def run(self):
        # 初始化算法
        self.init_algorithm()
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
        self.pop = np.array([tour], dtype=int)
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        self.record()

    def get_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
