import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Algorithms.ALGORITHM import ALGORITHM


class ACO(ALGORITHM):
    def __init__(self, problem, num_pop=50, num_iter=100, alpha=1, beta=4, rho=0.2, q_value=100, show_mode=0):
        """
        蚁群算法 (蚁周模型 Ant-Cycle)
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小(蚁群大小)
        :param num_iter: 迭代次数
        :param alpha: 信息素因子，反映信息素的重要程度，一般取值[1~4]
        :param beta: 启发函数因子，反映了启发式信息的重要程度，一般取值[3~5]
        :param rho: 信息素挥发因子，一般取值[0.1~0.5]
        :param q_value: 信息素常量，一般取值[10, 1000]
        :param show_mode: 绘图模式
        """
        super().__init__(problem, num_pop, num_iter, None, None, None, show_mode)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q_value = q_value
        self.dist_mat = None
        self.eta_mat = None
        self.tau_mat = None

    def init_algorithm(self):
        # 问题必须为单目标问题
        if self.problem.num_obj > 1:
            raise ValueError("This method can only solve single objective problems")
        # 问题必须为序列问题
        if np.sum(self.problem_type != ALGORITHM.PMU):
            raise ValueError("This method can only solve sequence problems")
        # 问题必须提供距离矩阵
        if not hasattr(self.problem, 'dist_mat'):
            raise ValueError("The problem must provide the distance matrix")
        # 初始化参数
        super().init_algorithm()
        # 获取问题的距离矩阵
        self.dist_mat = self.problem.dist_mat  # type: ignore
        # 调整距离矩阵的对角线元素值
        np.fill_diagonal(self.dist_mat, 1e-6)
        # 启发式信息，一般取距离的倒数
        self.eta_mat = 1 / self.dist_mat
        # 调整启发式信息对角线元素的值为0
        np.fill_diagonal(self.eta_mat, 0)
        # 路径上的信息素矩阵，初始化为1
        self.tau_mat = np.ones((self.num_dec, self.num_dec))
        # 蚁群路径(路径记录表, 记录已经访问过的节点)
        self.pop = np.zeros((self.num_pop, self.num_dec), dtype=int)

    @ALGORITHM.record_time
    def run(self):
        """运行算法(主函数)"""
        # 初始化算法
        self.init_algorithm()
        # 绘制初始状态图
        self.plot(pause=True, n_iter=0)
        for i in self.iterator:
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(pause=True, n_iter=i + 1)

    def run_step(self, i):
        # 清空路径记录表
        self.pop = np.zeros((self.num_pop, self.num_dec), dtype=int)
        # 随机生成各个蚂蚁的起点
        start_node = np.random.randint(self.num_dec, size=self.num_pop)
        # 将第一列赋值为当前的起点
        self.pop[:, 0] = start_node
        # 获取所有蚂蚁当前的行动路线
        for j in range(1, self.num_dec):
            # 获取当前起点对应的信息素和启发式信息情况
            tau_mat_ = self.tau_mat[self.pop[:, j - 1]]
            eta_mat_ = self.eta_mat[self.pop[:, j - 1]]
            # 对访问过的节点进行mask
            tau_mat_[np.arange(self.num_pop), self.pop[:, 0:j].T] = 0
            eta_mat_[np.arange(self.num_pop), self.pop[:, 0:j].T] = 0
            # 根据信息素和启发式信息计算下个节点的访问概率
            prob_mat = tau_mat_ ** self.alpha * eta_mat_ ** self.beta
            # 对访问概率按行(每个蚂蚁个体)归一化
            prob_mat = prob_mat / np.sum(prob_mat, -1)[:, np.newaxis]
            # 根据概率矩阵随机选择下标，来选出应该访问哪个节点作为下个节点
            chosen_indices = np.apply_along_axis(lambda row: np.random.choice(self.num_dec, p=row),
                                                 axis=1, arr=prob_mat)
            # 将产生的下个节点加入访问表，更新蚁群路径
            self.pop[:, j] = chosen_indices
        # 更新路径长度
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        # 重新计算等价适应度值
        self.fitness = self.get_fitness(self.objs, self.cons)
        # 更新信息素矩阵
        delta_tau_mat = np.zeros((self.num_dec, self.num_dec))
        # 使用add.at函数更新delta_tau_mat
        np.add.at(delta_tau_mat, (self.pop.flatten(), np.roll(self.pop, shift=-1, axis=1).flatten()),
                  np.repeat(self.q_value / self.fitness, self.num_dec))
        self.tau_mat = (1 - self.rho) * self.tau_mat + delta_tau_mat
        # 记录每步状态
        self.record(i + 1)

    def get_best(self):
        """覆写获取最优解，这里获取的是历史最优解"""
        best, best_obj, best_con = self.get_best_(self.pop, self.objs, self.cons)
        # 若满足约束则指定约束为0
        best_con = best_con if best_con > 0 else 0
        # 若解更满足约束或者目标值更好则更新解
        if (best_con < self.best_con) or (best_con == self.best_con and best_obj < self.best_obj):
            self.best, self.best_obj, self.best_con = best, best_obj, best_con

    def plot_(self, pause=False, n_iter=None, pause_time=0.1):
        """绘制优化过程中信息素变化情况"""
        if n_iter == 0:
            return
        if not pause:
            plt.figure()
        if not hasattr(self.problem, 'points'):
            raise ValueError("The drawing must provide the location of the points!")
        points = self.problem.points  # type: ignore
        num_points = len(points)
        # 获取权重非零的行和列索引
        rows, cols = np.nonzero(self.tau_mat)
        weights = self.tau_mat[rows, cols]
        # 对权重归一化（透明度必须在[0,1]之间）
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        plt.clf()
        graph = nx.Graph()
        # 添加节点信息
        graph.add_nodes_from(np.arange(num_points))
        edges_with_weights = list(zip(rows, cols, weights))
        graph.add_weighted_edges_from(edges_with_weights)
        # 点的位置
        pos = dict(zip(range(num_points), points))
        # 获取边的权重并设置透明度
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        edge_alphas = list(edge_weights.values())  # 权重已经在0到1之间，可以直接用作透明度
        # 控制点的大小
        if num_points >= 100:
            node_size = 50 / (num_points // 50)
        else:
            node_size = 100
        # 画图
        if pause and n_iter is not None:
            plt.title("iter: " + str(n_iter))
        # 绘制节点
        nx.draw_networkx_nodes(graph, pos, node_size=node_size)
        # 绘制边，并设置边的透明度
        nx.draw_networkx_edges(graph, pos, edge_color='black', alpha=edge_alphas, width=2)
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()
