# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions
from ....utilities.strategies import local_search_2opt


class LocalSearch(Algorithm):
    def __init__(self, zero_start: bool = False, visual_mode: Optional[str] = None):
        """
        局部搜索(Local Search)求解TSP问题 (使用2-opt算子)

        Code Maintainer: LuChen Wang
        :param zero_start: 得到的解是否从0开始
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=0, visual_mode=visual_mode)
        self.zero_start = zero_start
        self.single_obj_only = True
        self.supported_var_types = [self.PMU]
        self.dist_mat = None
        self.symmetric = True

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
        # 检查是否是非对称矩阵
        if hasattr(self.problem, 'symmetric'):
            self.symmetric = self.problem.symmetric

    def init_solutions(self, *args, **kwargs):
        # 计算TSP问题的节点数量
        num_nodes = len(self.dist_mat)
        # 初始化路由
        route = np.arange(num_nodes)
        # 进行局部搜索
        route = local_search_2opt(route, self.dist_mat, symmetric=self.symmetric)
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
