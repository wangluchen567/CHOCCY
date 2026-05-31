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
from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions
from scipy.spatial import distance_matrix
from ....utilities.commons import dom_matrix


class SPEA2(Algorithm):
    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 visual_mode: Optional[str] = None):
        """
        改进的基于优势排序的帕累托进化算法

        Reference Papers:
            SPEA2: Improving the strength Pareto evolutionary algorithm,
            E. Zitzler, M. Laumanns, and L. Thiele
        Code Maintainers:
            LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)

    def run_step(self, i):
        """运行算法单步"""
        # 选择阶段：从当前种群中选择父代个体组成配对池
        parent_indices = self.get_mating_indices()
        # 衍生阶段：对配对池中个体应用交叉和变异生成子代
        offspring = self.apply_operator(parent_indices)
        # 环境选择阶段：合并父代与子代，选择下一代种群
        self.environmental_selection(offspring)

    def eval_fits(self, sols: Solutions):
        """覆写评估解集的适应度值向量函数"""
        # 计算带约束惩罚的目标值(若有约束)
        objs = self.eval_penalized_objs(sols)
        # 得到每对解的支配关系
        dom_mat = dom_matrix(objs)
        # 得到 每个个体i支配的个体数 S
        s_values = np.sum(dom_mat, axis=1)
        # 得到 支配i的每个个体j支配的所有个体数之和 R
        r_values = np.zeros(sols.n_sols)
        for i in range(sols.n_sols):
            r_values[i] = np.sum(s_values[dom_mat[:, i] == 1])
        # 当多个个体不相互支配时需要使用k邻近估算密度
        # 计算每个个体目标值之间的距离
        dist_mat = distance_matrix(objs, objs)
        np.fill_diagonal(dist_mat, np.inf)  # 对角线设置为inf
        # 将距离按照递增排序并选第k=sqrt(N+N)个作为指标(N+N:父代+子代)
        dist_sort = np.sort(dist_mat, axis=1)
        d_values = 1.0 / (dist_sort[:, int(np.sqrt(sols.n_sols))] + 2)
        # 计算个体适应度值
        fits = r_values + d_values
        return fits

    def apply_selection(self, next_size: int):
        """覆写为SPEA2的选择策略进行选择"""
        # 初始化要选择的个体
        chosen = np.array(self.sols.fits < 1)
        num_chosen = np.sum(chosen)
        if num_chosen < self.n_sols:
            # 默认可选数量过少则进行补充
            ranking = np.argsort(self.sols.fits)
            chosen[ranking[:self.n_sols]] = True
        elif num_chosen > self.n_sols:
            # 若可选数量过多则进行裁剪
            # 为了能求解约束问题这里对根据约束计算的新目标值进行计算
            objs = self.eval_penalized_objs(self.sols)
            del_indices = self.truncation(objs[chosen], num_chosen - self.n_sols)
            chosen_indices = np.where(chosen)[0]
            chosen[chosen_indices[del_indices]] = False
        else:
            pass
        return chosen

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
