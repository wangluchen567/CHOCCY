from Algorithms.ALGORITHM import ALGORITHM
from Algorithms.Utility.Operators import operator_rand_de, operator_best_de


class DE(ALGORITHM):
    RAND1 = 3  # (DE/rand/1)
    RAND2 = 5  # (DE/rand/2)
    BEST1 = 2  # (DE/best/1)
    BEST2 = 4  # (DE/best/2)

    def __init__(self, num_pop=None, num_iter=None, cross_prob=None, mutate_prob=None,
                 factor=0.5, operator_type=BEST1, show_mode=0):
        """
        差分进化算法
        *Code Author: Luchen Wang
        :param num_pop: 种群大小
        :param num_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param factor: 缩放因子
        :param operator_type: 算子类型
        :param show_mode: 绘图模式
        """
        super().__init__(num_pop, num_iter, cross_prob, mutate_prob, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL, self.INT]
        self.factor = factor
        self.operator_type = operator_type
        self.num_parents = operator_type  # trick
        self.cross_prob = 0.9 if cross_prob is None else cross_prob

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取交配池
        mating_pools = [self.mating_pool_selection(self.num_pop) for _ in range(self.num_parents)]
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
