# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions
from ....utilities.strategies import fast_local_search_2opt


class GuidedFastLocalSearch(Algorithm):
    def __init__(self,
                 max_iter: int = 1000,
                 alpha: float = 0.25,
                 max_stagnation: int = 1000,
                 zero_start: bool = False,
                 visual_mode: Optional[str] = None):
        """
        引导式快速局部搜索(Local Search)求解TSP问题 (使用2-opt算子)

        Code Maintainer: LuChen Wang
        :param max_iter: 迭代次数
        :param alpha: 控制惩罚系数的超参数
        :param max_stagnation: 最大停滞次数限制
        :param zero_start: 得到的解是否从0开始
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=max_iter, visual_mode=visual_mode)
        self.alpha = alpha
        self.max_stagnation = max_stagnation
        self.zero_start = zero_start
        self.single_obj_only = True
        self.supported_var_types = [self.PMU]
        self.dist_mat = None
        self.symmetric = True
        self.bits = None
        self.pena_mat = None
        self.pena_coef = None
        self.route = None
        self.route_cost = None

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

    def prepare(self):
        # 设置当前解为单个解
        self.sols = self.sols[0]
        # 获取当前路由状态
        self.route = self.sols.x.astype(int)
        # 获取当前路由成本
        self.route_cost = self.sols.f
        # 初始化子邻域
        self.bits = np.ones(len(self.dist_mat))
        # 初始化惩罚矩阵与惩罚参数
        self.pena_mat = np.zeros_like(self.dist_mat)
        self.pena_coef = 0.0
        # 初始化停滞计数器
        self.stagnation = 0

    def run_step(self, i):
        # 进行快速局部搜索
        self.route = fast_local_search_2opt(self.bits, self.route, self.dist_mat,
                                            self.pena_mat, self.pena_coef,
                                            symmetric=self.symmetric)
        # 计算当前路由的成本
        self.route_cost = self.problem.calc_obj(self.route)
        # 计算效用值矩阵
        util_mat = self.dist_mat / (1 + self.pena_mat)
        # 获取边对应的效用值矩阵
        edge_utils = np.zeros_like(self.dist_mat)
        route_roll = np.roll(self.route, -1)
        edge_utils[route_roll, self.route] = util_mat[route_roll, self.route]
        edge_utils[self.route, route_roll] = util_mat[self.route, route_roll]
        util_max = edge_utils.max()  # 获取边对应的效用值矩阵的最大值
        # 更新惩罚矩阵与惩罚参数
        self.pena_mat[np.where(util_mat == util_max)] += 1
        self.pena_coef = self.alpha * self.route_cost / (len(self.route) + 1)
        # 激活端点子邻域
        self.bits = np.zeros_like(self.bits, dtype=bool)
        self.bits[np.argwhere(util_mat == util_max).ravel()] = True
        # 更新搜索到的最优解
        if self.route_cost < self.sols.objs[0][0]:
            self.sols.xs[:, :] = self.route.copy()
            # 对新解进行重新评估并更新最优解信息
            self.evaluate_and_update()
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _should_stop(self):
        return self.stagnation >= self.max_stagnation

    def finalize(self, sols: Solutions, inplace: bool = False) -> Solutions:
        """对解进行输出前的最终处理"""
        sols_ = super().finalize(sols, inplace)
        if self.zero_start:  # 将回路滚动为从0开始
            zero_index = np.where(sols_.x == 0)[0][0]
            sols_.xs[:, :] = np.roll(sols_.x, -zero_index)
        return sols_