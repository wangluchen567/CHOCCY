# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
import networkx as nx
from typing import Optional
from ....core import warn_once
from ...algorithm import Algorithm
from ....utilities.visualization import Frame


class ACO(Algorithm):
    def __init__(self,
                 n_sols: int = 50,
                 max_iter: int = 200,
                 alpha: float = 1.0,
                 beta: float = 3.0,
                 rho: float = 0.2,
                 q_const: float = 100.0,
                 visual_mode: Optional[str] = None):
        """
        蚁群算法 (蚁周模型 Ant-Cycle)

        Code Maintainer: LuChen Wang
        :param n_sols: 蚁群大小
        :param max_iter: 迭代次数
        :param alpha: 信息素因子，反映信息素的重要程度，一般取值[1 ~ 4]
        :param beta: 启发函数因子，反映了启发式信息的重要程度，一般取值[1 ~ 5]
        :param rho: 信息素挥发因子，一般取值[0.1 ~ 0.5]
        :param q_const: 信息素常量，一般取值[10 ~ 1000]
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, None, None, visual_mode)
        self.alpha = alpha  # 信息素因子
        self.beta = beta  # 启发函数因子
        self.rho = rho  # 信息素挥发因子
        self.q_const = q_const  # 信息素常量
        self.symmetric = True  # 是否是对称矩阵
        self.dist_mat = None  # 距离矩阵
        self.eta_mat = None  # 启发式信息矩阵
        self.tau_mat = None  # 信息素矩阵
        # 该蚁群优化算法仅支持 单目标的 TSP 问题优化
        self.single_obj_only = True
        self.supported_var_types = [self.PMU]

    def init_parameters(self):
        """初始化算法参数"""
        super().init_parameters()
        # 定义需要的属性
        if hasattr(self.problem, 'dist_mat'):
            # 所有必需属性都存在
            self.dist_mat = np.asarray(self.problem.dist_mat)
        else:
            raise AttributeError(
                f"Problem '{type(self.problem).__name__}' missing required 'dist_mat' attribute. "
                f"This algorithm requires a distance matrix. "
            )
        # 检查是否是非对称矩阵
        if hasattr(self.problem, 'symmetric'):
            self.symmetric = self.problem.symmetric
        # 调整距离矩阵的对角线元素值
        np.fill_diagonal(self.dist_mat, 1e-6)
        # 启发式信息，一般取距离的倒数
        self.eta_mat = np.asarray(1 / self.dist_mat)
        # 调整启发式信息对角线元素的值为 0
        np.fill_diagonal(self.eta_mat, 0)
        # 路径上的信息素矩阵，初始化为 1
        self.tau_mat = np.ones_like(self.dist_mat)

    def run_step(self, i):
        """运行算法单步"""
        # 清空蚁群路径记录表
        ants_path = np.zeros((self.n_sols, self.problem.n_vars), dtype=int)
        # 随机生成各个蚂蚁的起点
        start_node = np.random.randint(self.problem.n_vars, size=self.n_sols)
        # 将第一列赋值为当前的起点
        ants_path[:, 0] = start_node
        # 获取所有蚂蚁当前的行动路线
        for j in range(1, self.problem.n_vars):
            # 获取当前起点对应的信息素和启发式信息情况
            tau_mat_ = self.tau_mat[ants_path[:, j - 1]]
            eta_mat_ = self.eta_mat[ants_path[:, j - 1]]
            # 对访问过的节点进行mask
            tau_mat_[np.arange(self.n_sols), ants_path[:, 0:j].T] = 0
            eta_mat_[np.arange(self.n_sols), ants_path[:, 0:j].T] = 0
            # 根据信息素和启发式信息计算下个节点的访问概率
            prob_mat = tau_mat_ ** self.alpha * eta_mat_ ** self.beta
            # 对访问概率按行(每个蚂蚁个体)归一化
            prob_mat = prob_mat / np.sum(prob_mat, -1)[:, np.newaxis]
            # 根据概率矩阵随机选择下标，来选出应该访问哪个节点作为下个节点
            chosen_indices = np.apply_along_axis(lambda row: np.random.choice(self.problem.n_vars, p=row),
                                                 axis=1, arr=prob_mat)
            # 将产生的下个节点加入访问表，更新蚁群路径
            ants_path[:, j] = chosen_indices
        # 对种群进评估并更新相关参数
        self.sols.xs = ants_path
        # 对初始解集进行评估并更新最优解
        self.evaluate_and_update()
        # 初始化信息素更新矩阵
        delta_tau_mat = np.zeros_like(self.tau_mat)
        # 使用add.at函数更新delta_tau_mat
        np.add.at(delta_tau_mat, (ants_path.flatten(), np.roll(ants_path, shift=-1, axis=1).flatten()),
                  np.repeat(self.q_const / self.sols.fits, self.problem.n_vars))
        # 更新信息素矩阵
        self.tau_mat = (1 - self.rho) * self.tau_mat + delta_tau_mat

    def update_best(self):
        """更新最优解(覆写为根据全局规则更新最优解)"""
        self.update_best_global()

    def plot_by_algorithm(self,
                          n_iter: Optional[int] = None,
                          **kwargs):
        """覆写算法可视化函数（可视化信息素矩阵）"""
        if not hasattr(self.problem, 'locations'):
            warn_once("Missing attribute: 'locations'. "
                      "Please ensure your problem has locations defined.")
            return None
        if self.problem.locations is None:
            warn_once("'locations' is None. "
                      "Please initialize problem with Problem(locations=your_data).")
            return None
        # 初始化绘制的帧
        frame = Frame()
        # 创建要绘制的图
        graph = nx.Graph()
        # 计算节点个数
        num_nodes = len(self.tau_mat)
        # 获取权重非零的行和列索引
        rows, cols = np.nonzero(self.tau_mat)
        weights = self.tau_mat[rows, cols]
        # 对权重归一化（透明度必须在[0, 1]之间）
        if weights.max() == weights.min():
            weights = np.zeros_like(weights)
        else:
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        # 添加节点信息
        graph.add_nodes_from(np.arange(num_nodes))
        edges_with_weights = list(zip(rows, cols, weights))
        graph.add_weighted_edges_from(edges_with_weights)
        # 点的位置
        pos = dict(zip(range(num_nodes), self.problem.locations))
        # 获取边的权重并设置透明度
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        edge_alphas = list(edge_weights.values())  # 权重已经在0到1之间，可以直接用作透明度
        # 控制点的大小
        node_size = 100 if num_nodes < 100 else 50 / (num_nodes // 50)
        # 绘制信息素结果图
        frame.add_nx_nodes(graph, pos, node_size=node_size)
        frame.add_nx_edges(graph, pos, edge_color='black', alpha=edge_alphas, width=2)
        # 设置标题
        if n_iter is None:
            frame.set_title("Pheromone matrix")
        else:
            frame.set_title(f"Pheromone matrix (Iteration {n_iter})")
        return frame

    def get_config(self) -> dict:
        """获取算法的完整配置"""
        config = super().get_config()
        config['alpha'] = self.alpha
        config['beta'] = self.beta
        config['rho'] = self.rho
        config['q_const'] = self.q_const
        return config
