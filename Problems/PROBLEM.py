import numpy as np
from typing import Union

class PROBLEM(object):
    # 定义问题常量
    REAL = 0
    INT = 1
    BIN = 2
    PMU = 3
    FIX = 4

    def __init__(self,
                 problem_type: int,
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
        :param lower: 变量下界(包含下界)
        :param upper: 变量上界(包含上界)(整数问题不包含)
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
        # 覆写函数检查，只能覆写指定的函数
        if type(self).cal_objs != PROBLEM.cal_objs:
            raise TypeError("Method 'cal_objs' cannot be overridden, please overwrite method '_cal_objs'")
        if type(self).cal_cons != PROBLEM.cal_cons:
            raise TypeError("Method 'cal_cons' cannot be overridden, please overwrite method '_cal_cons'")

    def format_range(self):
        """重整决策变量取值范围"""
        if isinstance(self.lower, int) or isinstance(self.lower, float):
            self.lower = np.zeros(self.num_dec) + self.lower
        if isinstance(self.upper, int) or isinstance(self.upper, float):
            self.upper = np.zeros(self.num_dec) + self.upper
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
        if X_.ndim == 1:
            X_ = X_.reshape(1, -1)
        # 若为整数问题则需要向下取整
        if PROBLEM.INT in self.type_indices:
            X_[:, self.type_indices[PROBLEM.INT]] \
                = np.floor(X_[:, self.type_indices[PROBLEM.INT]])
        objs = self._cal_objs(X_)
        if objs.ndim == 1:
            return objs.reshape(-1, 1)
        else:
            return objs

    def cal_cons(self, X):
        """计算约束值"""
        # 对数据进行浅拷贝，防止其被修改
        X_ = X.copy()
        # 保证二维形状方便并行操作
        if X_.ndim == 1:
            X_ = X_.reshape(1, -1)
        # 若为整数问题则需要向下取整
        if PROBLEM.INT in self.type_indices:
            X_[:, self.type_indices[PROBLEM.INT]] \
                = np.floor(X_[:, self.type_indices[PROBLEM.INT]])
        cons = self._cal_cons(X_)
        if cons.ndim == 1:
            return cons.reshape(-1, 1)
        else:
            return cons

    def _cal_objs(self, X):
        """计算目标值(必须覆写)"""
        raise NotImplemented

    def _cal_cons(self, X):
        """计算约束值(默认无约束)(可覆写)"""
        return -np.ones(len(X))

    def get_optimum(self, *args, **kwargs):
        """获取理论最优目标值"""
        pass

    def get_pareto_front(self, *args, **kwargs):
        """获取帕累托最优前沿"""
        pass

    def plot_(self, *args, **kwargs):
        """问题提供的绘图函数"""
        pass
