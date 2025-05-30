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
from Algorithms import ALGORITHM
from Algorithms.Utility.Operators import operator_rand_de, operator_best_de


class DE(ALGORITHM):
    RAND1 = 3  # (DE/rand/1)
    RAND2 = 5  # (DE/rand/2)
    BEST1 = 2  # (DE/best/1)
    BEST2 = 4  # (DE/best/2)

    def __init__(self, pop_size=None, max_iter=None, cross_prob=None, mutate_prob=None,
                 factor=0.5, operator_type=BEST1, show_mode=0):
        """
        差分进化算法

        Code Author: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param factor: 缩放因子
        :param operator_type: 算子类型
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL, self.INT]
        self.factor = factor
        self.operator_type = operator_type
        self.num_parents = operator_type  # trick
        self.cross_prob = 0.9 if cross_prob is None else cross_prob

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取匹配池
        mating_pools = [self.mating_pool_selection(self.pop_size) for _ in range(self.num_parents)]
        # 交叉变异生成子代
        offspring = self.operator(mating_pools)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record()

    def operator(self, mating_pools):
        """重写算子为差分进化算子"""
        parents = [self.pop[mating_pool] for mating_pool in mating_pools]
        if self.operator_type == self.RAND1 or self.operator_type == self.RAND2:
            return operator_rand_de(parents, self.lower, self.upper, self.cross_prob, self.mutate_prob, self.factor)
        elif self.operator_type == self.BEST1 or self.operator_type == self.BEST2:
            return operator_best_de(parents, self.best, self.lower, self.upper,
                                    self.cross_prob, self.mutate_prob, self.factor)
        else:
            raise ValueError(f"The operator type {self.operator_type} does not exist")

    def environmental_selection(self, offspring):
        """差分进化环境选择"""
        # 先计算子代目标值与约束值
        off_objs = self.cal_objs(offspring)
        off_cons = self.cal_cons(offspring)
        # 计算子代的适应度值
        off_fits = self.cal_fits(off_objs, off_cons)
        # 得到更优的算子下标
        better = off_fits < self.fits
        # 将种群个体替换为优秀子代
        self.pop[better] = offspring[better]
        self.objs[better] = off_objs[better]
        self.cons[better] = off_cons[better]
        self.fits[better] = off_fits[better]

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        types = ['', '', 'DE/best/1', 'DE/rand/1', 'DE/best/2', 'DE/rand/2']
        info['factor'] = self.factor
        info['operator_type'] = types[self.operator_type]
        return info
