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
from Algorithms.Utility.Selections import elitist_selection
from Algorithms.Utility.Operators import operator_real, operator_binary


class NNDREAS(ALGORITHM):
    def __init__(self, pop_size=None, max_iter=None, structure=None, search_range=None, delta=0.5,
                 cross_prob=None, mutate_prob=None, show_mode=0):
        """
        基于神经网络降维的大规模二进制优化算法

        References:
            Neural Network-Based Dimensionality Reduction for LargeScale Binary Optimization With Millions of Variables,
            Ye Tian, Luchen Wang, Shangshang Yang, Jinliang Ding, Yaochu Jin, Xingyi Zhang
        Code Author:
            Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param structure: 神经网络结构
        :param search_range: 权重搜索范围
        :param delta: 第一阶段搜索占比
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.BIN]
        self.structure = structure
        self.search_range = search_range
        self.delta = delta
        # 初始化参数
        self.instance = None
        self.shapes = None
        self.pop_weights = None
        self.delta_iter = None

    def init_algorithm(self, problem, pop=None):
        super().init_algorithm(problem, pop)
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
        self.shapes = []
        self.num_dec = 0
        for i in range(len(self.structure) - 1):
            self.shapes.append([self.structure[i], self.structure[i + 1]])
            self.shapes.append([self.structure[i + 1]])
            self.num_dec += self.structure[i] * self.structure[i + 1]
            self.num_dec += self.structure[i + 1]
        # 由于问题转换为了实数问题，所以需要重新初始化算法相关参数
        if self.search_range is None:
            self.search_range = np.array([-100, 100])
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
        # 按照delta占比分为两个阶段
        self.delta_iter = self.delta * self.max_iter

    def init_pop_weights(self):
        pop_weights = np.random.uniform(self.upper, self.lower, size=(self.pop_size, self.num_dec))
        return pop_weights

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        if i <= self.delta_iter:
            # 获取匹配池
            mating_pool = self.mating_pool_selection()
            # 交叉变异生成子代
            offspring_weights = self.operator_weights(mating_pool)
            # 进行环境选择
            self.environmental_selection_weights(offspring_weights)
        else:
            # 获取匹配池
            mating_pool = self.mating_pool_selection()
            # 交叉变异生成子代
            offspring = self.operator_origin(mating_pool)
            # 进行环境选择
            self.environmental_selection(offspring)
        # 记录每步状态
        self.record()

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
        best_indices = elitist_selection(new_fits, self.pop_size)
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
        for i in range(len(self.shapes)):
            if len(self.shapes[i]) > 1:
                weight = weights[:, pointer: pointer + self.shapes[i][0] * self.shapes[i][1]]
                pointer = pointer + self.shapes[i][0] * self.shapes[i][1]
                output = np.matmul(output, weight.reshape(num_weights, self.shapes[i][0], self.shapes[i][1]))
            else:
                bias = weights[:, pointer: pointer + self.shapes[i][0]]
                pointer = pointer + self.shapes[i][0]
                output = output + bias.reshape(num_weights, 1, -1).repeat(ins_size, 1)
                if i == len(self.shapes) - 1:
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

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        info['structure'] = self.structure if isinstance(self.structure, list) else list(self.structure)
        info['search_range'] = self.search_range if isinstance(self.search_range, list) else list(self.search_range)
        info['delta'] = self.delta
        return info
