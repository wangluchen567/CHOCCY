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
import warnings
import numpy as np
from typing import Union


class PROBLEM(object):
    # 定义问题常量
    REAL = 0  # 实数
    INT = 1  # 整数
    BIN = 2  # 二进制
    PMU = 3  # 序列
    FIX = 4  # 固定标签

    def __init__(self,
                 problem_type: Union[int, np.ndarray],
                 num_dec: int,
                 num_obj: int,
                 lower: Union[float, np.ndarray],
                 upper: Union[float, np.ndarray]):
        """
        问题父类
        *Code Author: LuChen Wang
        :param problem_type: 问题类型 (0:实数, 1:整数, 2:二进制, 3:序列, 4:固定标签)(可混合)
        :param num_dec: 决策变量个数(维度)
        :param num_obj: 目标个数
        :param lower: 决策变量下界(包含下界)(可混合)
        :param upper: 决策变量上界(包含上界)(整数问题不包含)(可混合)
        """
        self.problem_type = problem_type  # 问题类型
        self.num_dec = num_dec  # 决策变量个数(维度)
        self.num_obj = num_obj  # 目标个数
        self.lower = lower  # 变量下界(包含下界)
        self.upper = upper  # 变量上界(包含上界)(整数问题不包含)
        self.unique_type = None  # 问题类型情况(混合情况)
        self.type_indices = None  # 每个问题类别对应的位置
        self.format_type()  # 重整问题类型与位置情况(混合情况)
        self.format_range()  # 重整变量上下界(额外处理整数问题)
        self.optimums = self.get_optimum()  # 获取理论最优目标值
        self.pareto_front = self.get_pareto_front()  # 获取帕累托最优前沿
        self.overwrite_cons = False  # 是否 计算约束的方法 至少有一个被覆写
        # 覆写函数检查，只能覆写指定的函数
        if type(self).cal_objs != PROBLEM.cal_objs:
            raise TypeError("Method 'cal_objs' cannot be overridden, please overwrite method '_cal_objs'")
        if type(self).cal_cons != PROBLEM.cal_cons:
            raise TypeError("Method 'cal_cons' cannot be overridden, please overwrite method '_cal_cons'")
        if type(self)._cal_objs != PROBLEM._cal_objs and type(self)._cal_obj != PROBLEM._cal_obj:
            warnings.warn("Both '_cal_objs' and '_cal_obj' have been overridden, "
                          "the current calculation is based on method '_cal_objs'")
        if type(self)._cal_cons != PROBLEM._cal_cons and type(self)._cal_con != PROBLEM._cal_con:
            warnings.warn("Both '_cal_cons' and '_cal_con' have been overridden, "
                          "the current calculation is based on method '_cal_cons'")
        if type(self)._cal_objs == PROBLEM._cal_objs and type(self)._cal_obj == PROBLEM._cal_obj:
            raise TypeError("At least one of methods '_cal_objs' or '_cal_obj' must be overridden")
        if type(self)._cal_cons == PROBLEM._cal_cons and type(self)._cal_con == PROBLEM._cal_con:
            self.overwrite_cons = False  # 计算约束的方法 均没有被覆写
        else:
            self.overwrite_cons = True  # 计算约束的方法 至少有一个被覆写
        # 检查计算单个解的目标函数梯度是否被覆写
        if type(self)._cal_obj_grad == PROBLEM._cal_obj_grad:
            self.overwrite_obj_grad = False
        else:
            self.overwrite_obj_grad = True
        # 检查计算单个解的约束函数梯度是否被覆写
        if type(self)._cal_con_grad == PROBLEM._cal_con_grad:
            self.overwrite_con_grad = False
        else:
            self.overwrite_con_grad = True

    def format_range(self):
        """重整决策变量取值范围"""
        if isinstance(self.lower, np.ndarray):
            assert self.lower.ndim == 1
            assert self.lower.shape[0] == self.num_dec
            self.lower = self.lower.astype(float)
        else:
            self.lower = np.zeros(self.num_dec, dtype=float) + self.lower
        if isinstance(self.upper, np.ndarray):
            assert self.upper.ndim == 1
            assert self.upper.shape[0] == self.num_dec
            self.upper = self.upper.astype(float)
        else:
            self.upper = np.zeros(self.num_dec, dtype=float) + self.upper
        # 整数问题不包含上界
        if PROBLEM.INT in self.type_indices:
            self.upper[self.type_indices[PROBLEM.INT]] -= 1e-9

    def format_type(self):
        """重整问题类型并确定混合位置"""
        if isinstance(self.problem_type, int):
            self.problem_type = np.zeros(self.num_dec, dtype=int) + self.problem_type
        self.unique_type = np.unique(self.problem_type)
        # 确定每个问题类别对应的位置
        self.type_indices = dict()
        for t in self.unique_type:
            self.type_indices[t] = np.where(self.problem_type == t)[0]

    def cal_objs(self, X):
        """计算目标值"""
        # 对数据进行浅拷贝，防止其被修改
        X_ = X.copy()
        # 保证二维形状方便并行操作
        X_ = X_.reshape(1, -1) if X_.ndim == 1 else X_
        # 若为整数问题则需要向下取整
        if PROBLEM.INT in self.type_indices:
            X_[:, self.type_indices[PROBLEM.INT]] = np.floor(X_[:, self.type_indices[PROBLEM.INT]])
        objs = self._cal_objs(X_)
        # 保证二维形状方便并行操作
        return objs.reshape(-1, 1) if objs.ndim == 1 else objs

    def cal_cons(self, X):
        """计算约束值"""
        # 对数据进行浅拷贝，防止其被修改
        X_ = X.copy()
        # 保证二维形状方便并行操作
        X_ = X_.reshape(1, -1) if X_.ndim == 1 else X_
        # 若为整数问题则需要向下取整
        if PROBLEM.INT in self.type_indices:
            X_[:, self.type_indices[PROBLEM.INT]] = np.floor(X_[:, self.type_indices[PROBLEM.INT]])
        cons = self._cal_cons(X_)
        # 保证二维形状方便并行操作
        return cons.reshape(-1, 1) if cons.ndim == 1 else cons

    def cal_grad(self, X):
        """
        计算梯度值
        :param X: 决策向量
        :return: 若无约束则返回目标函数梯度，否则返回 (目标函数梯度, 约束函数梯度)
        """
        # 对数据进行浅拷贝，防止其被修改
        X_ = X.copy()
        # 保证二维形状方便并行操作
        X_ = X_.reshape(1, -1) if X_.ndim == 1 else X_
        # 计算目标函数的梯度
        objs_grad = self._cal_objs_grad(X_)
        # 若约束计算未被覆写则无需计算约束函数梯度
        if not self.overwrite_cons:
            return objs_grad.reshape(-1, 1) if objs_grad.ndim == 1 else objs_grad
        else:  # 否则若约束计算被覆写则需额外计算约束函数梯度
            cons_grad = self._cal_cons_grad(X_)
            return (objs_grad.reshape(-1, 1) if objs_grad.ndim == 1 else objs_grad,
                    cons_grad.reshape(-1, 1) if cons_grad.ndim == 1 else cons_grad)

    def _cal_objs(self, X):
        """计算整个种群的目标值(建议覆写)"""
        pop_size = len(X)
        objs = np.zeros((pop_size, self.num_obj))
        for i in range(pop_size):
            objs[i] = self._cal_obj(X[i])
        return objs

    def _cal_cons(self, X):
        """计算整个种群的约束值(默认无约束)(可覆写)"""
        # 计算约束的方法 均没有被覆写 则使用默认值
        if not self.overwrite_cons:
            return -np.ones(len(X))
        pop_size = len(X)
        cons = np.zeros((pop_size, self.num_obj))
        for i in range(pop_size):
            cons[i] = self._cal_con(X[i])
        return cons

    def _cal_objs_grad(self, X):
        """计算整个种群目标函数的梯度"""
        if self.overwrite_obj_grad:
            # 若子类覆写了计算单个解的梯度则使用子类的计算
            pop_size = len(X)
            objs_grad = np.zeros((pop_size, self.num_dec))
            for i in range(pop_size):
                objs_grad[i] = self._cal_obj_grad(X[i])
            return objs_grad
        # 否则默认使用有限差分法估计目标函数梯度
        # obj_grad = f(x + epsilon) - f(x) / (x * epsilon)
        X[X == 0] = 1.e-12
        # 初始化目标函数值
        objs = self.cal_objs(X)
        # 并对该结果进行变换
        objs_trans = np.tile(objs, (self.num_obj, 1))
        # 对决策向量进行扰动
        X_disturb = np.tile(X, (self.num_dec, 1)) * (1 + np.eye(self.num_dec) * 1.e-6)
        # 得到扰动后的向量对应的目标值
        objs_disturb = self.cal_objs(X_disturb)
        # 使用有限差分法得到目标函数的梯度
        objs_grad = (objs_disturb - objs_trans).T / X / 1.e-6
        # 返回目标函数梯度结果
        return objs_grad

    def _cal_cons_grad(self, X):
        """计算整个种群约束函数的梯度"""
        if self.overwrite_obj_grad:
            # 若子类覆写了计算单个解的梯度则使用子类的计算
            pop_size = len(X)
            cons_grad = np.zeros((pop_size, self.num_dec))
            for i in range(pop_size):
                cons_grad[i] = self._cal_con_grad(X[i])
            return cons_grad
        # 否则默认使用有限差分法估计约束函数梯度
        # con_grad = g(x + epsilon) - g(x) / (x * epsilon)
        X[X == 0] = 1.e-12
        # 初始化约束函数值
        cons = self.cal_cons(X)
        # 并对该结果进行变换
        cons_trans = np.tile(cons, (self.num_obj, 1))
        # 对决策向量进行扰动
        X_disturb = np.tile(X, (self.num_dec, 1)) * (1 + np.eye(self.num_dec) * 1.e-6)
        # 得到扰动后的向量对应的目标值
        cons_disturb = self.cal_cons(X_disturb)
        # 使用有限差分法得到约束函数的梯度
        cons_grad = (cons_disturb - cons_trans).T / X / 1.e-6
        # 返回约束函数梯度结果
        return cons_grad

    def _cal_obj(self, x):
        """计算单个决策向量的目标值(可覆写)"""
        pass

    def _cal_con(self, x):
        """计算单个决策向量的约束值(可覆写)"""
        return -1

    def _cal_obj_grad(self, x):
        """计算单个决策向量的目标函数梯度"""
        return 0

    def _cal_con_grad(self, x):
        """计算单个决策向量的约束函数梯度"""
        return 0

    def get_optimum(self, *args, **kwargs):
        """获取理论最优目标值(或参考点向量)(形状必须为(N*M))"""
        return None

    def get_pareto_front(self, *args, **kwargs):
        """获取帕累托最优前沿(以绘图)"""
        return None

    def plot(self, *args, **kwargs):
        """问题提供的绘图函数"""
        pass

    def get_info(self):
        """获取问题的相关信息"""
        return {
            'problem_type': self.problem_type.tolist(),
            'num_dec': self.num_dec,
            'num_obj': self.num_obj,
            'lower': self.lower.tolist(),
            'upper': self.upper.tolist(),
            'unique_type': self.unique_type.tolist(),
        }
