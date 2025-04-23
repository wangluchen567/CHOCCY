"""
交叉算子
Crossover Operator

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


def simulated_binary_crossover(parents1, parents2, lower, upper, cross_prob, eta=20):
    """
    模拟二进制交叉(实数问题)
    :param parents1: 父代种群1
    :param parents2: 父代种群2
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param eta: 超参数
    :return: 子代种群
    """
    if parents1.shape != parents2.shape:
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    beta = np.zeros((p_size, num_dec))
    mu = np.random.random((p_size, num_dec))
    beta[mu <= 0.5] = np.power((2 * mu[mu <= 0.5]), 1 / (eta + 1))
    beta[mu > 0.5] = np.power((2 - 2 * mu[mu > 0.5]), -1 / (eta + 1))
    beta = beta * np.power(-1, np.random.randint(2, size=(p_size, num_dec)))
    beta[np.random.random((p_size, 1)).repeat(num_dec, 1) > cross_prob] = 1
    offspring = np.concatenate((
        (parents1 + parents2) / 2 + beta * (parents1 - parents2) / 2,
        (parents1 + parents2) / 2 - beta * (parents1 - parents2) / 2
    ))
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((2 * p_size, num_dec)) * lower
        uppers = np.ones((2 * p_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(2 * p_size, 0)
        uppers = upper.reshape(1, -1).repeat(2 * p_size, 0)
    offspring[offspring < lowers] = lowers[offspring < lowers]
    offspring[offspring > uppers] = uppers[offspring > uppers]
    return offspring


def binary_crossover(parents1, parents2, cross_prob):
    """
    二进制均匀交叉(二进制问题)
    :param parents1: 父代种群1
    :param parents2: 父代种群2
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parents1.shape != parents2.shape:
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    # 维度方面均匀交叉，个数方面按照交叉概率交叉
    mask = (np.random.rand(p_size, num_dec) < 0.5) & (np.random.rand(p_size, 1) < cross_prob)
    # 若mask为true则取第一个矩阵元素,否则取第二个矩阵中元素
    offspring1 = np.where(mask, parents2, parents1)
    offspring2 = np.where(mask, parents1, parents2)
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def order_crossover(parents1, parents2, cross_prob):
    """
    顺序交叉(序列问题)
    :param parents1: 父代种群1
    :param parents2: 父代种群2
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parents1.shape != parents2.shape:
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    offspring1 = parents1.copy()
    offspring2 = parents2.copy()
    # 生成所有需要的随机数
    crossover_mask = np.random.random(p_size) < cross_prob
    starts1 = np.random.randint(0, num_dec, size=p_size)
    ends1 = np.random.randint(starts1 + 1, num_dec + 1, size=p_size)
    starts2 = np.random.randint(0, num_dec, size=p_size)
    ends2 = np.random.randint(starts2 + 1, num_dec + 1, size=p_size)
    for i in range(p_size):
        if crossover_mask[i]:
            # 进行顺序交叉
            offspring1_ = list(dict.fromkeys(np.concatenate((parents1[i][starts1[i]:ends1[i]],
                                                             np.roll(parents2[i], -starts2[i])))))
            offspring1[i] = np.roll(np.array(offspring1_), starts1[i])
            offspring2_ = list(dict.fromkeys(np.concatenate((parents2[i][starts2[i]:ends2[i]],
                                                             np.roll(parents1[i], -starts1[i])))))
            offspring2[i] = np.roll(np.array(offspring2_), starts2[i])
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def fix_label_crossover(parents1, parents2, cross_prob):
    """
    固定类型数的标签的均匀交叉(固定类型数标签问题)
    :param parents1: 父代种群1
    :param parents2: 父代种群2
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parents1.shape[0] != parents2.shape[0]:
        raise ValueError("The size of the two parent populations is not equal")
    if parents1.shape[1] != parents2.shape[1]:
        raise ValueError("The dim of the two parent populations is not equal")
    # 得到每种标签的类型和数量
    labels_type, labels_num = np.unique(parents1[0], return_counts=True)
    offspring = fix_label_cx(parents1, parents2, labels_type, labels_num, cross_prob)
    return offspring


def fix_label_cx_(parents1, parents2, labels_type, labels_num, cross_prob):
    """
    固定类型数的标签的均匀交叉(子函数)
    :param parents1: 父代种群1
    :param parents2: 父代种群2
    :param labels_type: 每种标签的类型
    :param labels_num: 每种标签的数量
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    p_size, num_dec = parents1.shape
    # 初始化子代
    offspring1 = np.zeros(parents1.shape, dtype=int)
    offspring2 = np.zeros(parents2.shape, dtype=int)
    # 两父代相同位保持不变，不同位均匀交叉，并且需要保证标签等量约束
    equals = np.array(parents1 == parents2, dtype=bool)
    offspring1[equals] = parents1[equals]
    offspring2[equals] = parents2[equals]
    # 这里需要遍历以满足固定数量的约束
    for i in range(p_size):
        # 统计剩余标签数量
        last_labels1 = labels_num.copy()
        last_labels2 = labels_num.copy()
        for j in range(len(labels_type)):
            last_labels1[j] -= np.sum(offspring1[i] == labels_type[j])
            last_labels2[j] -= np.sum(offspring2[i] == labels_type[j])
        # 根据现存数量在考虑约束的情况下得到子代
        for j in range(num_dec):
            if equals[i][j]:
                pass
            else:
                # 随机从父代中选择继承点
                r1 = (parents1[i][j] if np.random.random() < 0.5 else parents2[i][
                    j]) if np.random.random() < cross_prob else offspring1[i][j]
                r2 = (parents2[i][j] if np.random.random() < 0.5 else parents1[i][
                    j]) if np.random.random() < cross_prob else offspring2[i][j]
                k1, k2 = np.where(labels_type == r1)[0], np.where(labels_type == r2)[0]
                # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                if last_labels1[k1] <= 0:
                    k1 = np.random.choice(np.where(last_labels1 > 0)[0])
                    r1 = labels_type[k1]
                offspring1[i][j] = r1
                last_labels1[k1] -= 1
                # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                if last_labels2[k2] <= 0:
                    k2 = np.random.choice(np.where(last_labels2 > 0)[0])
                    r2 = labels_type[k2]
                offspring2[i][j] = r2
                last_labels2[k2] -= 1
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def de_rand_1(offspring, parents1, parents2, lower, upper, cross_prob, factor):
    """
    差分进化算法随机算子1 (DE/rand/1)
    :param offspring: 要进行交叉的种群
    :param parents1: 差分的父代1
    :param parents2: 差分的父代2
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param factor: 缩放因子
    :return: 子代种群
    """
    if not (offspring.shape == parents1.shape == parents2.shape):
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    site = np.random.random((p_size, num_dec)) < cross_prob
    offspring[site] = offspring[site] + factor * (parents1[site] - parents2[site])
    # 上下界裁剪
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((p_size, num_dec)) * lower
        uppers = np.ones((p_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(p_size, 0)
        uppers = upper.reshape(1, -1).repeat(p_size, 0)
    offspring[offspring < lowers] = lowers[offspring < lowers]
    offspring[offspring > uppers] = uppers[offspring > uppers]
    return offspring


def de_rand_2(offspring, parents1, parents2, parents3, parents4, lower, upper, cross_prob, factor):
    """
    差分进化算法随机算子2 (DE/rand/2)
    :param offspring: 要进行交叉的种群
    :param parents1: 差分的父代1
    :param parents2: 差分的父代2
    :param parents3: 差分的父代3
    :param parents4: 差分的父代4
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param factor: 缩放因子
    :return: 子代种群
    """
    if not (offspring.shape == parents1.shape == parents2.shape == parents3.shape == parents4.shape):
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    site = np.random.random((p_size, num_dec)) < cross_prob
    offspring[site] = (offspring[site] + factor * (parents1[site] - parents2[site]) +
                       factor * (parents3[site] - parents4[site]))
    # 上下界裁剪
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((p_size, num_dec)) * lower
        uppers = np.ones((p_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(p_size, 0)
        uppers = upper.reshape(1, -1).repeat(p_size, 0)
    offspring[offspring < lowers] = lowers[offspring < lowers]
    offspring[offspring > uppers] = uppers[offspring > uppers]
    return offspring


def de_best_1(best, parents1, parents2, lower, upper, cross_prob, factor):
    """
    差分进化算法最优个体算子1 (DE/best/1)
    :param best: 父代中最优个体
    :param parents1: 差分的父代1
    :param parents2: 差分的父代2
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param factor: 缩放因子
    :return: 子代种群
    """
    if not (parents1.shape == parents2.shape):
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    site = np.random.random((p_size, num_dec)) < cross_prob
    offspring = np.repeat(best[np.newaxis, :], p_size, axis=0)
    offspring[site] = offspring[site] + factor * (parents1[site] - parents2[site])
    # 上下界裁剪
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((p_size, num_dec)) * lower
        uppers = np.ones((p_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(p_size, 0)
        uppers = upper.reshape(1, -1).repeat(p_size, 0)
    offspring[offspring < lowers] = lowers[offspring < lowers]
    offspring[offspring > uppers] = uppers[offspring > uppers]
    return offspring


def de_best_2(best, parents1, parents2, parents3, parents4, lower, upper, cross_prob, factor):
    """
    差分进化算法最优个体算子2 (DE/best/2)
    :param best: 父代中最优个体
    :param parents1: 差分的父代1
    :param parents2: 差分的父代2
    :param parents3: 差分的父代3
    :param parents4: 差分的父代4
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param factor: 缩放因子
    :return: 子代种群
    """
    if not (parents1.shape == parents2.shape):
        raise ValueError("The shape of the two parent populations is not equal")
    p_size, num_dec = parents1.shape
    site = np.random.random((p_size, num_dec)) < cross_prob
    offspring = np.repeat(best[np.newaxis, :], p_size, axis=0)
    offspring[site] = (offspring[site] + factor * (parents1[site] - parents2[site]) +
                       factor * (parents3[site] - parents4[site]))
    # 上下界裁剪
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((p_size, num_dec)) * lower
        uppers = np.ones((p_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(p_size, 0)
        uppers = upper.reshape(1, -1).repeat(p_size, 0)
    offspring[offspring < lowers] = lowers[offspring < lowers]
    offspring[offspring > uppers] = uppers[offspring > uppers]
    return offspring


try:
    # 尝试导入numba
    from numba import jit


    @jit(nopython=True, cache=True)
    def fix_label_cx_jit(parents1, parents2, labels_type, labels_num, cross_prob):
        """
        固定类型数的标签的均匀交叉(子函数)(Numba加速版本)
        :param parents1: 父代种群1
        :param parents2: 父代种群2
        :param labels_type: 每种标签的类型
        :param labels_num: 每种标签的数量
        :param cross_prob: 交叉概率
        :return: 子代种群
        """
        p_size, num_dec = parents1.shape
        # 初始化子代
        offspring1 = np.zeros(parents1.shape, dtype=np.int32)
        offspring2 = np.zeros(parents2.shape, dtype=np.int32)
        # 两父代相同位保持不变，不同位均匀交叉，并且需要保证标签等量约束
        # 使用显式循环代替布尔数组索引
        for i in range(p_size):
            for j in range(num_dec):
                if parents1[i, j] == parents2[i, j]:
                    offspring1[i, j] = parents1[i, j]
                    offspring2[i, j] = parents2[i, j]

        # 这里需要遍历以满足固定数量的约束
        for i in range(p_size):
            # 统计剩余标签数量
            last_labels1 = labels_num.copy()
            last_labels2 = labels_num.copy()
            for j in range(len(labels_type)):
                count1 = 0
                count2 = 0
                for k in range(num_dec):
                    if offspring1[i, k] == labels_type[j]:
                        count1 += 1
                    if offspring2[i, k] == labels_type[j]:
                        count2 += 1
                last_labels1[j] -= count1
                last_labels2[j] -= count2

            # 根据现存数量在考虑约束的情况下得到子代
            for j in range(num_dec):
                if parents1[i, j] != parents2[i, j]:
                    # 随机从父代中选择继承点
                    if np.random.random() < cross_prob:
                        r1 = parents1[i, j] if np.random.random() < 0.5 else parents2[i, j]
                    else:
                        r1 = offspring1[i, j]

                    if np.random.random() < cross_prob:
                        r2 = parents2[i, j] if np.random.random() < 0.5 else parents1[i, j]
                    else:
                        r2 = offspring2[i, j]

                    # 找到对应的标签类型索引
                    k1 = -1
                    k2 = -1
                    for idx in range(len(labels_type)):
                        if labels_type[idx] == r1:
                            k1 = idx
                        if labels_type[idx] == r2:
                            k2 = idx

                    # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                    if last_labels1[k1] <= 0:
                        available = np.where(last_labels1 > 0)[0]
                        if len(available) > 0:
                            k1 = np.random.choice(available)
                            r1 = labels_type[k1]

                    offspring1[i, j] = r1
                    last_labels1[k1] -= 1

                    # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                    if last_labels2[k2] <= 0:
                        available = np.where(last_labels2 > 0)[0]
                        if len(available) > 0:
                            k2 = np.random.choice(available)
                            r2 = labels_type[k2]

                    offspring2[i, j] = r2
                    last_labels2[k2] -= 1

        offspring = np.vstack((offspring1, offspring2))
        return offspring


    def fix_label_cx(parents1, parents2, labels_type, labels_num, cross_prob):
        """
        包装函数，保持原始接口不变，内部调用Numba加速版本
        """
        # 确保输入数组是Numba兼容的类型
        parents1 = np.asarray(parents1, dtype=np.int32)
        parents2 = np.asarray(parents2, dtype=np.int32)
        labels_type = np.asarray(labels_type, dtype=np.int32)
        labels_num = np.asarray(labels_num, dtype=np.int32)

        return fix_label_cx_jit(parents1, parents2, labels_type, labels_num, cross_prob)


except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Optimizing problems without using numba acceleration...")
    fix_label_cx = fix_label_cx_
