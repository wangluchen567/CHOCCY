"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import warnings
import numpy as np
from typing import Optional
from Algorithms import ALGORITHM


class GFLS(ALGORITHM):
    def __init__(self,
                 max_iter: Optional[int] = 1000,
                 alpha: float = 1 / 4,
                 active_all: bool = True,
                 show_mode: Optional[str] = None):
        """
        引导快速局部搜索(Guided Fast Local Search)

        References:
            GUIDED LOCAL SEARCH,
            Christos Voudouris, Edward P. K. Tsang and Abdullah Alsheddy
        Code Maintainer:
            Luchen Wang
        :param max_iter: 迭代次数
        :param alpha: 用于更新lambda的超参数
        :param active_all: 是否激活全部子邻域
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size=1, max_iter=max_iter, show_mode=show_mode)
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
    def init_algorithm(self, problem):
        super().init_algorithm(problem)
        # 获取问题的距离矩阵
        self.dist_mat = self.problem.dist_mat  # type: ignore
        # 初始化lambda值
        self.lamb = 0
        # 初始化惩罚矩阵
        self.p_mat = np.zeros(self.dist_mat.shape)

    @ALGORITHM.record_time
    def init_and_eval_pop(self, pop=None):
        super().init_and_eval_pop(pop)
        # 问题必须提供距离矩阵
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
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
            self.bits[np.argwhere(utils == utils_max).ravel()] = 1

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
        self.bits[np.argwhere(utils == utils_max).ravel()] = 1
        if self.tour_cost < self.objs[0]:
            self.pop[0] = self.tour.copy()
            self.eval_and_update(self.pop)
        # 记录每步状态
        self.record()

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        info['active_all'] = self.active_all
        info['alpha'] = self.alpha
        return info


def fast_local_search_(tour, dist_mat, bits, p_mat, lamb):
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
    idx = np.where(tour == i)[0][0]
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


try:
    # 尝试导入numba
    from numba import jit


    @jit(nopython=True, cache=True)
    def fast_local_search_jit(tour, dist_mat, bits, p_mat, lamb):
        """
        快速局部搜索
        :param tour: TSP回路
        :param dist_mat: 距离矩阵
        :param bits: 子邻域激活情况
        :param p_mat: 惩罚矩阵
        :param lamb: 超参数
        :return: 搜索结果
        """
        while np.sum(bits) > 0:
            # 搜索激活的子邻域
            for i in range(len(tour)):
                if bits[i]:
                    tour, improved, active_set = two_opt_jit(tour, dist_mat, p_mat, lamb, i)
                    if improved:  # 如果有提升，则激活其子邻域
                        bits[active_set] = 1
                    else:  # 否则冻结其子领域
                        bits[i] = 0
        # 返回搜索后的回路
        return tour


    def fast_local_search(tour, dist_mat, bits, p_mat, lamb):
        return fast_local_search_jit(tour, dist_mat, bits, p_mat, lamb)


    @jit(nopython=True, cache=True)
    def two_opt_jit(tour, dist_mat, p_mat, lamb, i):
        """
        搜索第i个节点的邻域
        :param tour: TSP回路
        :param dist_mat: 距离矩阵
        :param p_mat: 惩罚矩阵
        :param lamb: 超参数
        :param i: 第i个节点
        :return: 搜索后的结果
        """
        improved = False
        active_set = np.zeros(4, dtype=np.int32)
        idx = np.where(tour == i)[0][0]
        tour = np.concatenate((tour[idx:], tour[:idx]))
        for j in range(3, len(tour)):
            n1, n2, n3, n4 = tour[0], tour[1], tour[j - 1], tour[j]
            # 记录要激活的子邻域
            active_set = np.array([n1, n2, n3, n4])
            if cost_change_jit(dist_mat, p_mat, lamb, n1, n2, n3, n4) < -1e-9:
                tour[1:j] = tour[j - 1:0:-1]
                improved = True
                # 如果有提示则直接返回结果
                return tour, improved, active_set
        # 返回搜索后的回路，是否有提升，激活的子邻域
        return tour, improved, active_set


    @jit(nopython=True, cache=True)
    def cost_change_jit(dist_mat, p_mat, lamb, n1, n2, n3, n4):
        """
        计算带惩罚的2-opt后的收益值
        :param dist_mat: 距离矩阵
        :param p_mat: 惩罚矩阵
        :param lamb: 超参数
        :param n1: 2-opt的节点1
        :param n2: 2-opt的节点2
        :param n3: 2-opt的节点3
        :param n4: 2-opt的节点4
        :return: 收益值
        """
        result = dist_mat[n1, n3] + dist_mat[n2, n4] - dist_mat[n1, n2] - dist_mat[n3, n4]
        p = p_mat[n1, n3] + p_mat[n2, n4] - p_mat[n1, n2] - p_mat[n3, n4]
        total = result + lamb * p
        return total


except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Optimizing problems without using numba acceleration...")
    fast_local_search = fast_local_search_
