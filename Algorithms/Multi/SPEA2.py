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
import numpy as np
from Algorithms import ALGORITHM
from scipy.spatial import distance_matrix
from Algorithms.Utility.SupportUtils import get_dom_between


class SPEA2(ALGORITHM):
    def __init__(self, pop_size=None, max_iter=None, cross_prob=None, mutate_prob=None, show_mode=0):
        """
        改进的基于优势的帕累托进化算法

        References:
            SPEA2: Improving the strength Pareto evolutionary algorithm,
            E. Zitzler, M. Laumanns, and L. Thiele
        Code Author:
            Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, None, show_mode)

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取匹配池
        mating_pool = self.mating_pool_selection()
        # 交叉变异生成子代
        offspring = self.operator(mating_pool)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record()

    def cal_fits(self, objs, cons):
        """根据给定目标值和约束值得到适应度值"""
        pop_size = len(objs)
        # 检查是否均满足约束，若均满足约束则无需考虑约束
        if np.all(cons <= 0):
            objs_ = objs
        else:
            objs_ = self.cal_objs_based_cons(objs, cons)
        # 得到每对解的支配关系
        dom_between = get_dom_between(objs_)
        # 得到 每个个体i支配的个体数 S
        s_values = np.sum(dom_between, axis=1)
        # 得到 支配i的每个个体j支配的所有个体数之和 R
        r_values = np.zeros(pop_size)
        for i in range(pop_size):
            r_values[i] = np.sum(s_values[dom_between[:, i] == 1])
        # 当多个个体不相互支配时需要使用k邻近估算密度
        # 计算每个个体目标值之间的距离
        dist_mat = distance_matrix(objs_, objs_)
        np.fill_diagonal(dist_mat, np.inf)  # 对角线设置为inf
        # 将距离按照递增排序并选第k=sqrt(N+N)个作为指标(N+N:父代+子代)
        dist_sort = np.sort(dist_mat, axis=1)
        d_values = 1.0 / (dist_sort[:, int(np.sqrt(pop_size))] + 2)
        # 计算个体适应度值
        fits = r_values + d_values
        return fits

    def environmental_selection(self, offspring):
        """SPEA2环境选择"""
        # 将当前种群与其子代合并
        new_pop, new_objs, new_cons, new_fits = self.pop_merge(offspring)
        # 为了能求解约束问题这里对根据约束计算的新目标值进行计算
        new_objs_ = self.cal_objs_based_cons(new_objs, new_cons)
        # 使用SPEA2选择策略进行选择
        chosen = np.array(new_fits < 1)
        num_chosen = np.sum(chosen)
        if num_chosen < self.pop_size:
            # 默认可选数量过少则进行补充
            ranking = np.argsort(new_fits)
            chosen[ranking[:self.pop_size]] = True
        elif num_chosen > self.pop_size:
            # 若可选数量过多则进行裁剪
            del_indices = self.truncation(new_objs_[chosen], num_chosen - self.pop_size)
            chosen_indices = np.where(chosen)[0]
            chosen[chosen_indices[del_indices]] = False
        else:
            pass
        self.pop = new_pop[chosen]
        self.objs = new_objs[chosen]
        self.cons = new_cons[chosen]
        self.fits = new_fits[chosen]

    @staticmethod
    def truncation(objs, k):
        """
        截断选择(选择k个个体进行删除)

        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param objs: 种群的目标值向量
        :param k: 选择删除的个体数量
        :return: 个体是否被删除的标签向量
        """
        # 计算每个个体目标值之间的距离
        dist_mat = distance_matrix(objs, objs)
        np.fill_diagonal(dist_mat, np.inf)  # 对角线设置为inf
        # 初始化删除标志数组
        del_flag = np.zeros(objs.shape[0], dtype=bool)
        # 寻找要删除的个体
        while np.sum(del_flag) < k:
            # 找到尚未被删除的个体索引
            remain = np.where(~del_flag)[0]
            # 提取剩余个体之间的距离矩阵
            temp = dist_mat[np.ix_(remain, remain)]
            # 对每一行的距离进行排序，并获取排序后的索引
            sorted_indices = np.argsort(temp, axis=1)
            # 找到距离最小的个体索引
            min_index = sorted_indices[:, 1].min()
            # 将该个体标记为删除
            del_flag[remain[min_index]] = True
        return del_flag
