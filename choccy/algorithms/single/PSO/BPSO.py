# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Union, Optional
from ...algorithm import Algorithm
from ....utilities.commons import sigmoid
from ....utilities.visualization import plot_hybrids_2d


class BPSO(Algorithm):
    def __init__(self,
                 n_sols: int = 50,
                 max_iter: int = 200,
                 w: Union[float, tuple] = 1.0,
                 c1: float = 1.494,
                 c2: float = 1.494,
                 v_factor: float = 6.0,
                 visual_mode: Optional[str] = None):
        """
        粒子群优化算法（求解二进制问题版本）

        References:
            A Discrete Binary Version of The Particle Swarm Algorithm,
            James Kennedy and Russell C. Eberhar
        Code Maintainer:
            LuChen Wang
        :param n_sols: 粒子群大小
        :param max_iter: 迭代次数
        :param w: 惯性权重
        :param c1: 个体学习权重
        :param c2: 社会学习权重
        :param v_factor: 控制粒子速度的比例因子
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, None, None, visual_mode)
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习权重
        self.c2 = c2  # 社会学习权重
        self.v_factor = v_factor  # 控制粒子速度的比例因子
        self.particles = None  # 粒子群位置
        self.velocities = None  # 粒子群速度
        # 用于后续速度上下界裁剪
        self.v_min, self.v_max = None, None
        # 用于后续实现动态可变的惯性权重
        self.w_start, self.w_end = None, None
        # 仅支持 单目标的 二进制 问题优化
        self.single_obj_only = True
        self.supported_var_types = [self.BIN]

    def init_parameters(self):
        """初始化算法参数"""
        super().init_parameters()
        # 设置速度上下界，以方便后续用于裁剪
        self.v_min = self.v_factor * (self.problem.l_bounds - self.problem.u_bounds)
        self.v_max = self.v_factor * (self.problem.u_bounds - self.problem.l_bounds)
        # 初始化可变惯性权重
        if isinstance(self.w, tuple):
            if len(self.w) != 2:
                raise ValueError(
                    f"Inertia weight tuple must contain exactly 2 elements (start, end), "
                    f"but got {len(self.w)} elements: {self.w}"
                )
            self.w_start, self.w_end = float(self.w[0]), float(self.w[1])
        elif isinstance(self.w, float):
            self.w_start = self.w_end = self.w
        else:
            raise TypeError(
                f"Invalid type for inertia weight. Expected float or tuple (start, end), "
                f"but got {type(self.w).__name__}: {self.w}"
            )

    def prepare(self):
        # 初始化粒子群位置
        self.particles = self.sols.copy()
        # 初始化粒子群速度为随机值
        self.velocities = np.random.uniform(self.v_min, self.v_max, size=self.sols.xs.shape)

    def run_step(self, i):
        """运行算法单步"""
        # 动态调整惯性参数
        self.update_weight(i)
        # 优化得到新粒子群
        self.apply_operator_pso()
        # 更新粒子群个体最优位置
        self.update_particles()

    def update_weight(self, iteration):
        """动态调整惯性参数"""
        progress = iteration / self.max_iter
        self.w = self.w_start - (self.w_start - self.w_end) * progress

    def apply_operator_pso(self):
        """粒子群优化算子"""
        # 创建两个随机矩阵以引入随机性（学习因子）
        r1 = np.random.uniform(size=self.particles.xs.shape)
        r2 = np.random.uniform(size=self.particles.xs.shape)
        # 计算下一代粒子群速度（当前种群 sols 作为 p_best）
        self.velocities = (self.w * self.velocities +
                           r1 * self.c1 * (self.sols.xs - self.particles.xs) +
                           r2 * self.c2 * (self.best.xs - self.particles.xs))
        # 对粒子群速度进行裁剪
        self.velocities = np.clip(self.velocities, self.v_min, self.v_max)
        # 将速度转换为概率值
        probs = sigmoid(self.velocities)
        # 计算下一代粒子群位置（在二进制超立方体中）
        self.particles.xs = (np.random.uniform(size=probs.shape) < probs).astype(int)

    def update_particles(self):
        """更新粒子群个体最优位置"""
        # 使用一对一竞争更新粒子群个体最优位置
        self.local_selection(self.particles)

    def update_best(self):
        """更新最优解(覆写为根据全局规则更新最优解)"""
        self.update_best_global()

    def plot_by_algorithm(self,
                          n_iter: Optional[int] = None,
                          **kwargs):
        """覆写算法可视化函数（可视化粒子群的位置）"""
        # 必须使用copy函数否则无效果
        decs = self.particles.xs.copy()
        objs = self.particles.objs.copy()
        frame = plot_hybrids_2d(self.problem, decs, objs, n_iter, **self.visual_config)
        return frame
