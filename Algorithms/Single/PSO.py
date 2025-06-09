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


class PSO(ALGORITHM):
    def __init__(self, pop_size=None, max_iter=None, w=0.7298, c1=1.5, c2=1.5, show_mode=0):
        """
        粒子群优化算法

        Code Author: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param w: 惯性权重
        :param c1: 个体学习权重
        :param c2: 社会学习权重
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, None, None, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL, self.INT]
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习权重
        self.c2 = c2  # 社会学习权重
        self.particle = None  # 粒子群位置
        self.velocity = None  # 粒子群速度

    def init_algorithm(self, problem, pop=None):
        super().init_algorithm(problem, pop)
        # 初始化粒子群位置和速度
        self.particle = self.pop.copy()
        self.velocity = np.zeros_like(self.particle)

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 优化得到新粒子群
        self.operator_pso()
        # 更新粒子群个体最优位置
        self.update_particle()
        # 记录每步状态
        self.record()

    def operator_pso(self):
        """重写算子为粒子群优化算子"""
        pop_size, num_dec = self.pop.shape
        # 得到两个随机矩阵以引入随机性
        r1 = np.random.uniform(size=(pop_size, num_dec))
        r2 = np.random.uniform(size=(pop_size, num_dec))
        # 计算下一代粒子群速度
        self.velocity = (self.w * self.velocity +
                         r1 * self.c1 * (self.pop - self.particle) +
                         r2 * self.c2 * (self.best - self.particle))
        # 计算下一代粒子群位置
        self.particle = self.particle + self.velocity
        # 对上下界进行裁剪
        if isinstance(self.lower, int) or isinstance(self.lower, float):
            lower_ = np.ones((pop_size, num_dec)) * self.lower
            upper_ = np.ones((pop_size, num_dec)) * self.upper
        else:
            lower_ = self.lower.reshape(1, -1).repeat(pop_size, 0)
            upper_ = self.upper.reshape(1, -1).repeat(pop_size, 0)
        self.particle[self.particle < lower_] = lower_[self.particle < lower_]
        self.particle[self.particle > upper_] = upper_[self.particle > upper_]

    def update_particle(self):
        """更新粒子群个体最优位置"""
        # 计算目标值、约束值和适应度值
        particle_objs = self.cal_objs(self.particle)
        particle_cons = self.cal_cons(self.particle)
        particle_fits = self.cal_fits(particle_objs, particle_cons)
        # 得到更优的个体下标
        better = particle_fits < self.fits
        # 更新个体最优位置
        self.pop[better] = self.particle[better]
        self.objs[better] = particle_objs[better]
        self.cons[better] = particle_cons[better]
        self.fits[better] = particle_fits[better]

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        info['w'] = self.w
        info['c1'] = self.c1
        info['c2'] = self.c2
        return info
