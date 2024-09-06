import numpy as np


def simulated_binary_crossover(parent1, parent2, lower, upper, cross_prob, eta=20):
    """
    模拟二进制交叉(实数问题)
    :param parent1: 父代种群1
    :param parent2: 父代种群2
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param eta: 超参数
    :return: 子代种群
    """
    if parent1.shape[0] != parent2.shape[0]:
        raise ValueError("The size of the two parent populations is not equal")
    if parent1.shape[1] != parent2.shape[1]:
        raise ValueError("The dim of the two parent populations is not equal")
    N, D = parent1.shape
    beta = np.zeros((N, D))
    mu = np.random.random((N, D))
    beta[mu <= 0.5] = np.power((2 * mu[mu <= 0.5]), 1 / (eta + 1))
    beta[mu > 0.5] = np.power((2 - 2 * mu[mu > 0.5]), -1 / (eta + 1))
    beta = beta * np.power(-1, np.random.randint(2, size=(N, D)))
    beta[np.random.random((N, 1)).repeat(D, 1) > cross_prob] = 1
    offspring = np.concatenate((
        (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
        (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    ))
    if isinstance(lower, int) or isinstance(lower, float):
        Lower = np.ones((2 * N, D)) * lower
        Upper = np.ones((2 * N, D)) * upper
    else:
        Lower = lower.reshape(1, -1).repeat(2 * N, 0)
        Upper = upper.reshape(1, -1).repeat(2 * N, 0)
    offspring[offspring < Lower] = Lower[offspring < Lower]
    offspring[offspring > Upper] = Upper[offspring > Upper]
    return offspring


def binary_crossover(parent1, parent2, cross_prob):
    """
    二进制均匀交叉(二进制问题)
    :param parent1: 父代种群1
    :param parent2: 父代种群2
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parent1.shape[0] != parent2.shape[0]:
        raise ValueError("The size of the two parent populations is not equal")
    if parent1.shape[1] != parent2.shape[1]:
        raise ValueError("The dim of the two parent populations is not equal")
    N, D = parent1.shape
    # 维度方面均匀交叉，个数方面按照交叉概率交叉
    mask = (np.random.rand(N, D) < 0.5) & (np.random.rand(N, 1) < cross_prob)
    # 若mask为true则取第一个矩阵元素,否则取第二个矩阵中元素
    offspring1 = np.where(mask, parent2, parent1)
    offspring2 = np.where(mask, parent1, parent2)
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def order_crossover(parent1, parent2, cross_prob):
    """
    顺序交叉(序列问题)
    :param parent1: 父代种群1
    :param parent2: 父代种群2
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parent1.shape[0] != parent2.shape[0]:
        raise ValueError("The size of the two parent populations is not equal")
    if parent1.shape[1] != parent2.shape[1]:
        raise ValueError("The dim of the two parent populations is not equal")
    N, D = parent1.shape
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    for i in range(N):
        if np.random.random() < cross_prob:
            parent1_ = parent1[i]
            parent2_ = parent2[i]
            # 选择交叉片段
            point1 = np.random.randint(0, len(parent1_))
            point2 = np.random.randint(point1 + 1, len(parent1_) + 1)
            # 进行拼接
            temp1 = list(np.concatenate((parent1_[point1:point2], parent2_)))
            temp2 = list(np.concatenate((parent2_[point1:point2], parent1_)))
            # 去重后按原来元素重排
            offspring1_ = list(set(temp1))
            offspring1_.sort(key=temp1.index)
            offspring1[i] = np.array(offspring1_)
            offspring2_ = list(set(temp2))
            offspring2_.sort(key=temp2.index)
            offspring2[i] = np.array(offspring2_)
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def fix_label_crossover(parent1, parent2, cross_prob):
    """
    固定类型数的标签的均匀交叉(固定类型数标签问题)
    :param parent1: 父代种群1
    :param parent2: 父代种群2
    :param lower: 标签取值范围的整型下界
    :param upper: 标签取值范围的整型上界
    :param cross_prob: 交叉概率
    :return: 子代种群
    """
    if parent1.shape[0] != parent2.shape[0]:
        raise ValueError("The size of the two parent populations is not equal")
    if parent1.shape[1] != parent2.shape[1]:
        raise ValueError("The dim of the two parent populations is not equal")
    N, D = parent1.shape
    # 得到每种标签的类型和数量
    labels_type, labels_num = np.unique(parent1[0], return_counts=True)
    # 初始化子代
    offspring1 = np.zeros(parent1.shape, dtype=int)
    offspring2 = np.zeros(parent2.shape, dtype=int)
    # 两父代相同位保持不变，不同位均匀交叉，并且需要保证标签等量约束
    equals = np.array(parent1 == parent2, dtype=bool)
    offspring1[equals] = parent1[equals]
    offspring2[equals] = parent2[equals]
    # 这里需要遍历以满足固定数量的约束
    for i in range(N):
        # 统计剩余标签数量
        last_labels1 = labels_num.copy()
        last_labels2 = labels_num.copy()
        for j in range(len(labels_type)):
            last_labels1[j] -= np.sum(offspring1[i] == labels_type[j])
            last_labels2[j] -= np.sum(offspring2[i] == labels_type[j])
        # 根据现存数量在考虑约束的情况下得到子代
        for j in range(D):
            if equals[i][j]:
                pass
            else:
                # 随机从父代中选择继承点
                r1 = (parent1[i][j] if np.random.random() < 0.5 else parent2[i][
                    j]) if np.random.random() < cross_prob else offspring1[i][j]
                r2 = (parent2[i][j] if np.random.random() < 0.5 else parent1[i][
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
