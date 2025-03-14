import numpy as np
from tqdm import tqdm
from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Selections import elitist_selection
from Algorithms.Utility.Operators import operator_real, operator_binary
from Algorithms.Utility.Utils import fast_nd_sort, cal_crowd_dist, cal_ranking


class NNDREA(ALGORITHM):
    def __init__(self, problem, num_pop=100, num_iter=100, structure=None, search_range=None, delta=0.5,
                 cross_prob=None, mutate_prob=None, show_mode=0):
        """
        This code is based on the research presented in
        "Neural Network-Based Dimensionality Reduction for Large-Scale Binary Optimization With Millions of Variables"
        by Ye Tian, Luchen Wang, Shangshang Yang, Jinliang Ding, Yaochu Jin, Xingyi Zhang
        *Code Author: Luchen Wang
        :param problem: 问题对象
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param structure: 神经网络结构
        :param search_range: 权重搜索范围
        :param delta: 第一阶段搜索占比
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        # 初始化相关参数(调用父类初始化)
        super().__init__(problem, num_pop, num_iter, cross_prob, mutate_prob, None, show_mode)
        self.solvable_type = [self.BIN]
        self.structure = structure
        self.search_range = search_range
        self.delta = delta
        # 初始化参数
        self.instance = None
        self.slist = None
        self.pop_weights = None
        self.delta_iter = None

    @ALGORITHM.record_time
    def init_algorithm(self, pop=None):
        """初始化算法"""
        # 问题必须提供实例数据集
        if not hasattr(self.problem, 'instance'):
            raise ValueError("The problem must provide an instance dataset")
        # 实例数据集的大小必须与问题大小相同
        if (self.problem.instance.shape[0] != self.problem.num_dec and  # type: ignore
                self.problem.instance.shape[1] != self.problem.num_dec):  # type: ignore
            raise ValueError("The size of the instance dataset must be the same as the size of the problem")
        # 若实例大小在第二维度则转置以方便矩阵运算
        if self.problem.instance.shape[1] == self.problem.num_dec:  # type: ignore
            self.instance = self.problem.instance.T  # type: ignore
        else:
            self.instance = self.problem.instance  # type: ignore
        # 对实例中相同的数据进行扰动防止相同数据输出同一个值
        # self.instance += np.random.normal(0, 0.1, self.instance.shape)
        # 需要提供神经网络的结构信息(否则默认为[D, 4, 1])
        if self.structure is None:
            self.structure = [self.instance.shape[1], 4, 1]
        # 根据结构信息得到结构列表和权重数量以方便计算
        self.slist = []
        self.num_dec = 0
        for i in range(len(self.structure) - 1):
            self.slist.append([self.structure[i], self.structure[i + 1]])
            self.slist.append([self.structure[i + 1]])
            self.num_dec += self.structure[i] * self.structure[i + 1]
            self.num_dec += self.structure[i + 1]
        # 由于问题转换为了实数问题，所以需要重新初始化算法相关参数
        if self.search_range is None:
            self.search_range = np.array([-1, 1])
        self.lower = self.search_range[0] + np.zeros(self.num_dec)
        self.upper = self.search_range[1] + np.zeros(self.num_dec)
        # 初始化交叉和变异概率
        self.cross_prob = 1.0
        self.mutate_prob = 1 / self.num_dec
        # 初始化权重种群，计算目标值和约束值
        self.pop_weights = self.init_pop_weights()
        # 经过神经网络权重计算得到真实种群，并计算目标值
        self.pop = self.model_forward(self.pop_weights).astype(int)
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        self.fits = self.cal_fits(self.objs, self.cons)
        # 记录当前种群信息
        self.record()
        # 构建迭代器
        self.iterator = tqdm(range(self.num_iter)) if self.show_mode == 0 else range(self.num_iter)
        # 按照delta占比分为两个阶段
        self.delta_iter = self.delta * self.num_iter

    def init_pop_weights(self):
        pop_weights = np.random.uniform(self.upper, self.lower, size=(self.num_pop, self.num_dec))
        return pop_weights

    def run(self):
        """运行算法(主函数)"""
        # 初始化算法
        self.init_algorithm()
        # 绘制初始状态图
        self.plot(n_iter=0, pause=True)
        for i in self.iterator:
            # 运行单步算法
            self.run_step(i)
            # 绘制迭代过程中每步状态
            self.plot(n_iter=i + 1, pause=True)

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        if i <= self.delta_iter:
            # 获取交配池
            mating_pool = self.mating_pool_selection()
            # 交叉变异生成子代
            offspring_weights = self.operator_weights(mating_pool)
            # 进行环境选择
            self.environmental_selection_weights(offspring_weights)
        else:
            # 获取交配池
            mating_pool = self.mating_pool_selection()
            # 交叉变异生成子代
            offspring = self.operator_origin(mating_pool)
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

    def operator_weights(self, mating_pool):
        # 进行交叉变异生成子代
        return operator_real(self.pop_weights[mating_pool], self.lower, self.upper, self.cross_prob, self.mutate_prob)

    def operator_origin(self, mating_pool):
        # 进行交叉变异生成子代
        return operator_binary(self.pop[mating_pool], self.cross_prob, 1 / self.problem.num_dec)

    def environmental_selection_weights(self, offspring_weights):
        """对权重进行环境选择"""
        # 先计算子代目标值
        offspring = self.model_forward(offspring_weights).astype(int)
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 将父代与子代合并获得新种群
        new_pop = np.vstack((self.pop, offspring))
        new_objs = np.vstack((self.objs, off_objs))
        new_cons = np.vstack((self.cons, off_cons))
        new_pop_weights = np.vstack((self.pop_weights, offspring_weights))
        # 重新计算合并种群的的等价适应度值
        new_fits = self.cal_fits(new_objs, new_cons)
        # 使用选择策略(默认精英选择)选择进入下一代新种群的个体
        best_indices = elitist_selection(new_fits, self.num_pop)
        # 取目标值最优的个体组成新的种群
        self.pop = new_pop[best_indices]
        self.objs = new_objs[best_indices]
        self.cons = new_cons[best_indices]
        self.fits = new_fits[best_indices]
        self.pop_weights = new_pop_weights[best_indices]

    def model_forward(self, weights):
        """神经网络映射函数"""
        num_weights = len(weights)
        ins_size = len(self.instance)
        output = np.array([self.instance]).repeat(num_weights, 0)
        pointer = 0
        for i in range(len(self.slist)):
            if len(self.slist[i]) > 1:
                weight = weights[:, pointer: pointer + self.slist[i][0] * self.slist[i][1]]
                pointer = pointer + self.slist[i][0] * self.slist[i][1]
                output = np.matmul(output, weight.reshape(num_weights, self.slist[i][0], self.slist[i][1]))
            else:
                bias = weights[:, pointer: pointer + self.slist[i][0]]
                pointer = pointer + self.slist[i][0]
                output = output + bias.reshape(num_weights, 1, -1).repeat(ins_size, 1)
                if i == len(self.slist) - 1:
                    output = self.step(output)
                else:
                    output = self.leaky_relu(output)
        return output.squeeze()

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def leaky_relu(x, tiny=0.01):
        return x * (x > 0) + tiny * x * (x <= 0)

    @staticmethod
    def step(x):
        y = np.zeros(x.shape)
        y[x > 0] = 1.0
        y[x <= 0] = 0.0
        return y
