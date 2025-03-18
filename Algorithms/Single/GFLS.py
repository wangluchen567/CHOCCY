import numpy as np
from Algorithms.ALGORITHM import ALGORITHM


class GFLS(ALGORITHM):
    def __init__(self, problem, num_iter=1000, alpha=1 / 4, active_all=True, show_mode=0):
        """
        引导快速局部搜索(Guided Fast Local Search)
        *Code Author: Luchen Wang
        :param problem: 问题对象(TSP类型)
        :param num_iter: 迭代次数
        :param alpha: 用于更新lambda的超参数
        :param active_all: 是否激活全部子邻域
        :param show_mode: 绘图模式
        """
        super().__init__(problem, 1, num_iter, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.PMU]
        self.alpha = alpha
        self.active_all = active_all
        self.dist_mat = None
        self.p_mat = None
        self.lamb = None
        self.bits = None
        self.tour = None
        self.tour_cost = None

    @ALGORITHM.record_time
    def init_algorithm(self, pop=None):
        # 问题必须提供距离矩阵
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
        # 初始化参数
        super().init_algorithm(pop)
        # 获取问题的距离矩阵
        self.dist_mat = self.problem.dist_mat  # type: ignore
        # 初始化lambda值
        self.lamb = 0
        # 初始化惩罚矩阵
        self.p_mat = np.zeros(self.dist_mat.shape)
        # 获取当前路由状态
        self.tour = self.pop[0].astype(int)
        # 初始化子领域
        if self.active_all:
            # 开始激活所有子邻域
            self.bits = np.ones(len(self.tour))
        else:
            # 初始化效用矩阵
            utils = np.zeros(self.dist_mat.shape)
            tour_roll = np.concatenate((self.tour[-1:], self.tour[:-1]))
            # 根据当前状态计算效用矩阵
            utils[tour_roll, self.tour] = self.dist_mat[tour_roll, self.tour]
            utils[self.tour, tour_roll] = self.dist_mat[self.tour, tour_roll]
            # 得到最大效用边
            utils_max = np.max(utils)
            # 开始只激活最长边端点的子领域
            self.bits = np.zeros(len(self.tour))
            self.bits[np.where(utils == utils_max)[0][0]] = 1
            self.bits[np.where(utils == utils_max)[0][1]] = 1

    def run(self):
        """运行算法(主函数)"""
        # 初始化算法
        self.init_algorithm()
        # 绘制初始状态图
        self.plot(n_iter=0, pause=True)
        for i in self.iterator:
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(n_iter=i + 1, pause=True)

    @ALGORITHM.record_time
    def run_step(self, i):
        # 进行一次快速局部搜索
        self.tour = fast_local_search(self.tour, self.dist_mat, self.bits, self.p_mat, self.lamb)
        self.tour_cost = self.cal_objs(self.tour)[0][0]
        utils = np.zeros(self.dist_mat.shape)
        # 更新惩罚矩阵
        u_mat = self.dist_mat / (1 + self.p_mat)
        tour_roll = np.concatenate((self.tour[-1:], self.tour[:-1]))
        utils[tour_roll, self.tour] = u_mat[tour_roll, self.tour]
        utils[self.tour, tour_roll] = u_mat[self.tour, tour_roll]
        utils_max = np.max(utils)
        self.p_mat[np.where(utils == utils_max)] += 1
        # 更新lambda值
        self.lamb = self.alpha * self.tour_cost / (len(self.tour) + 1)
        # 激活端点子领域
        self.bits = np.zeros(len(self.tour))
        self.bits[np.where(utils == utils_max)[0][0]] = 1
        self.bits[np.where(utils == utils_max)[0][1]] = 1
        if self.tour_cost < self.objs[0]:
            self.pop[0] = self.tour.copy()
            self.objs[0] = self.tour_cost
            self.cons = self.cal_cons(self.pop)
        # 记录每步状态
        self.record()


def fast_local_search(tour, dist_mat, bits, p_mat, lamb):
    """快速局部搜索"""
    while np.sum(bits) > 0:
        for i in range(len(tour)):
            if bits[i]:
                tour, improved, active_set = two_opt(tour, dist_mat, p_mat, lamb, i)
                if improved:  # 如果有提升，则激活其子邻域
                    bits[active_set] = 1
                else:  # 否则冻结其子领域
                    bits[i] = 0
    return tour


def two_opt(tour, dist_mat, p_mat, lamb, i):
    """搜索第i个城市的邻域"""
    improved = False
    active_set = []
    idx = np.where(np.array(tour) == i)[0][0]
    tour = np.concatenate((tour[idx:], tour[:idx]))
    for j in range(3, len(tour)):
        n1, n2, n3, n4 = tour[0], tour[1], tour[j - 1], tour[j]
        # 记录要激活的子领域
        active_set = [n1, n2, n3, n4]
        if cost_change(dist_mat, p_mat, lamb, n1, n2, n3, n4) < -1e-9:
            tour[1:j] = tour[j - 1:0:-1]
            improved = True
            return tour, improved, active_set
    return tour, improved, active_set


def cost_change(dist_mat, p_mat, lamb, n1, n2, n3, n4):
    """计算带惩罚的2-opt后的收益值"""
    result = dist_mat[n1, n3] + dist_mat[n2, n4] - dist_mat[n1, n2] - dist_mat[n3, n4]
    p = p_mat[n1, n3] + p_mat[n2, n4] - p_mat[n1, n2] - p_mat[n3, n4]
    total = result + lamb * p
    return total
