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
from Algorithms.Utility.Operators import operator_diff


class DE(ALGORITHM):
    RAND_1 = 'DE/rand/1'
    RAND_2 = 'DE/rand/2'
    BEST_1 = 'DE/best/1'
    BEST_2 = 'DE/best/2'
    RAND_BEST_1 = 'DE/rand-to-best/1'
    CURRENT_BEST_1 = 'DE/current-to-best/1'

    def __init__(self, pop_size=None, max_iter=None, cross_prob=None, factor=None, operator_type=RAND_1, show_mode=0):
        """
        差分进化算法

        Code Maintainer: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param factor: 缩放因子
        :param operator_type: 算子类型
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, None, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL, self.INT]
        self.factor = factor
        self.operator_type = operator_type
        self.cross_prob = 0.2 if cross_prob is None else cross_prob
        self.factor = 0.5 if factor is None else factor

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取匹配池
        mating_pools = self.generate_pools()
        # 交叉变异生成子代
        offspring = self.operator(mating_pools)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record()

    def operator(self, mating_pools):
        """重写算子为差分进化算子"""
        parents = np.array([self.pop[mating_pool] for mating_pool in mating_pools])
        return operator_diff(self.pop, parents, self.lower, self.upper, self.cross_prob, self.factor)

    def environmental_selection(self, offspring):
        """差分进化环境选择"""
        # 先计算子代目标值与约束值
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 计算子代的适应度值
        off_fits = self.cal_fits(off_objs, off_cons)
        # 得到更优的算子下标
        better = off_fits <= self.fits
        # 将种群个体替换为优秀子代
        self.pop[better] = offspring[better]
        self.objs[better] = off_objs[better]
        self.cons[better] = off_cons[better]
        self.fits[better] = off_fits[better]

    def generate_pools(self):
        """根据算子设置生成所需要的个体下标"""
        if self.operator_type == self.RAND_1:
            # V = X_{r1} + F * (X_{r2} - X_{r3})
            indices = self.sample_indices(num_parents=3)
        elif self.operator_type == self.RAND_2:
            # V = X_{r1} + F * (X_{r2} - X_{r3}) + F * (X_{r4} - X_{r5})
            indices = self.sample_indices(num_parents=5)
        elif self.operator_type == self.BEST_1:
            # V = X_{best} + F * (X_{r1} - X_{r2})
            best_index = np.argmin(self.fits)  # 获取最优解下标
            indices = self.sample_indices(num_parents=2)
            indices.insert(0, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.BEST_2:
            # V = X_{best} + F * (X_{r1} - X_{r2}) + F * (X_{r3} - X_{r4})
            best_index = np.argmin(self.fits)  # 获取最优解下标
            indices = self.sample_indices(num_parents=4)
            indices.insert(0, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.RAND_BEST_1:
            # V = X_{r1} + F * (X_{best} - X_{r1}) + F * (X_{r2} - X_{r3})
            best_index = np.argmin(self.fits)  # 获取最优解下标
            indices = self.sample_indices(num_parents=3)
            indices.insert(0, indices[0])
            indices.insert(1, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.CURRENT_BEST_1:
            # V = X_i + F * (X_{best} - X_i) + F * (X_{r1} - X_{r2})
            best_index = np.argmin(self.fits)  # 获取最优解下标
            indices = self.sample_indices(num_parents=2)
            indices.insert(0, np.arange(self.pop_size))
            indices.insert(0, np.arange(self.pop_size))
            indices.insert(1, np.array([best_index] * self.pop_size))
        else:
            raise ValueError(f"There is no such operator type: {self.operator_type}")
        return indices

    def sample_indices(self, num_parents):
        """生成随机不重复下标(无法完全保证不重复)"""
        indices = []
        for _ in range(num_parents):
            indices.append(np.random.permutation(self.pop_size))
        return indices

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        info['factor'] = self.factor
        info['operator_type'] = self.operator_type
        return info
