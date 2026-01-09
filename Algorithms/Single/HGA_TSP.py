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
from typing import Optional
from Algorithms import ALGORITHM
from Algorithms.Utility.Educations import educate_tsp


class HGATSP(ALGORITHM):
    def __init__(self,
                 pop_size: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 educate_prob: Optional[float] = None,
                 show_mode: Optional[str] = None):
        """
        混合遗传算法(求解TSP问题)

        Code Maintainer: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param educate_prob: 教育概率
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, educate_prob, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.PMU]

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 选择阶段：从当前种群中选择父代个体组成配对池
        parent_indices = self.get_mating_indices()
        # 衍生阶段：对配对池中个体应用交叉和变异生成子代
        offspring = self.apply_operator(parent_indices)
        # 教育阶段：对子代进行教育（等价于局部搜索操作）
        offspring_educated = self.apply_education(offspring)
        # 环境选择阶段：合并父代与子代，选择下一代种群
        self.environmental_selection(offspring_educated)
        # 监控并记录每步状态
        self.record()

    def apply_education(self, offspring):
        """对子代进行教育"""
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
        return educate_tsp(self.problem.dist_mat, offspring, self.educate_prob)
