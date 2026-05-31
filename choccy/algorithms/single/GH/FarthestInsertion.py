# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions


class FarthestInsertion(Algorithm):
    def __init__(self, zero_start: bool = False, visual_mode: Optional[str] = None):
        """
        最远插入启发式算法(Farthest Insertion)求解TSP问题

        Code Maintainer: LuChen Wang
        :param zero_start: 得到的解是否从0开始
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=0, visual_mode=visual_mode)
        self.zero_start = zero_start
        self.single_obj_only = True
        self.supported_var_types = [self.PMU]
        self.dist_mat = None

    def init_parameters(self):
        super().init_parameters()
        # 定义需要的属性
        if hasattr(self.problem, 'dist_mat'):
            # 所有必需属性都存在
            self.dist_mat = self.problem.dist_mat
        else:
            raise AttributeError(
                f"Problem '{type(self.problem).__name__}' missing required 'dist_mat' attribute. "
                f"This algorithm requires a distance matrix. "
            )

    def init_solutions(self, *args, **kwargs):
        # 初始化路由
        route = []
        # 计算TSP问题的节点数量
        num_nodes = len(self.dist_mat)
        # 初始化节点是否已选的掩码
        mask = np.zeros(num_nodes, dtype=bool)
        # 逐步插入点构造解，时间复杂度 O(n)
        for i in range(num_nodes):
            # 得到候选下标
            cand_index = np.flatnonzero(mask == 0)
            if i == 0:
                # 找到最远的点
                chosen = self.dist_mat.max(axis=1).argmax()
                # 选择该点插入路由
                route = [chosen]
                # 已选的点进行mask
                mask[chosen] = True
            else:
                # 从距离已选点最近的候选点中选择距离最远的插入
                chosen = cand_index[self.dist_mat[np.ix_(~mask, mask)].min(axis=1).argmax()]
                # 计算插入成本
                insert_cost = (self.dist_mat[route, chosen] +
                               self.dist_mat[chosen, np.roll(route, -1)] -
                               self.dist_mat[route, np.roll(route, -1)])
                # 计算插入位置
                insert_index = np.argmin(insert_cost)
                # 在路由中该位置插入节点
                route.insert(insert_index + 1, chosen)
                # 已选的点进行mask
                mask[chosen] = True
        # 将list转换为array
        route = np.array(route, dtype=int)
        if self.zero_start:
            # 将回路滚动为从0开始
            zero_index = np.where(route == 0)[0][0]
            route = np.roll(route, -zero_index)
        # 得到最远插入算法的解
        self.sols = Solutions(decs=np.array([route], dtype=int))
        # 设置解集的评估函数
        self.set_evaluate_funcs()
        # 对初始解集进行评估并更新最优解
        self.evaluate_and_update()
