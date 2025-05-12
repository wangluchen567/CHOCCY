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
from Algorithms.Utility.SupportUtils import fast_nd_sort, cal_crowd_dist, cal_ranking


class NSGAII(ALGORITHM):
    def __init__(self, pop_size=None, max_iter=None, cross_prob=None, mutate_prob=None, show_mode=0):
        """
        基于快速非支配排序多目标遗传算法

        References:
            A fast and elitist multi-objective genetic algorithm: NSGA-II,
            K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan
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
        # 检查是否均满足约束，若均满足约束则无需考虑约束
        if np.all(cons <= 0):
            objs_based_cons = objs
        else:
            objs_based_cons = self.cal_objs_based_cons(objs, cons)
        # 对于多目标问题则需要考虑所在前沿面及拥挤度情况
        fronts, ranks = fast_nd_sort(objs_based_cons)
        crowd_dist = cal_crowd_dist(objs, fronts)
        fits = cal_ranking(ranks, crowd_dist)

        return fits
