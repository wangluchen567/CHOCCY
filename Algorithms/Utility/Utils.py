import itertools
import numpy as np


def is_dom(p_objs, q_objs):
    """
    定义支配关系
    :param p_objs: 个体p的目标值向量
    :param q_objs: 个体q的目标值向量
    :return: 个体p是否支配个体q
    """
    # 将输入转换为NumPy数组
    p_objs = np.array(p_objs)
    q_objs = np.array(q_objs)
    # 条件1: 对所有子目标， p 不比 q 差
    condition1 = np.all(p_objs <= q_objs)
    # 条件2: 至少存在一个子目标， p 比 q 好
    condition2 = np.any(p_objs < q_objs)
    # 满足以上两个条件则说明 p 支配 q
    return condition1 and condition2


def fast_nd_sort(objs):
    """快速非支配排序"""
    num_pop = len(objs)  # 获取种群数量
    objs = np.array(objs)  # 将目标转换为numpy数组
    fronts = []  # 初始化各前沿面列表
    ranks = np.zeros(num_pop, dtype=int)  # 每个个体所在的前沿数
    n = np.zeros(num_pop, dtype=int)  # 每一个解被支配的次数初始化为0
    S = [[] for _ in range(num_pop)]  # 每一个解所支配的解列表

    # 创建比较矩阵以确定支配关系
    for i in range(num_pop):
        # 判断解 i 是否支配其他解
        dominates = np.all(objs[i] <= objs, axis=1) & np.any(objs[i] < objs, axis=1)
        # 记录被解 i 支配的解
        S[i] = np.where(dominates)[0].tolist()
        # 更新每一个解被支配的次数
        n += dominates.astype(int)

    # 找出第一前沿面
    first_front = np.where(n == 0)[0].tolist()
    ranks[n == 0] = 1  # 将第一前沿面的排序序号设置为1
    fronts.append(first_front)  # 将第一前沿面添加到fronts列表中

    i = 0
    # 迭代处理每一个前沿面
    while i < len(fronts):
        Q = []  # 初始化下一个前沿面
        for p in fronts[i]:  # 遍历当前前沿面的每一个解
            for q in S[p]:  # 遍历每一个被当前解支配的解
                n[q] -= 1  # 被支配的次数减1
                if n[q] == 0:  # 如果支配次数减为0，表示该解进入下一前沿面
                    ranks[q] = i + 2  # 排序序号更新
                    Q.append(q)  # 将解添加到下一前沿面
        if Q:  # 如果下一个前沿面不为空
            fronts.append(Q)  # 将下一个前沿面添加到fronts列表中
        i += 1  # 处理下一个前沿面

    return fronts, ranks


def cal_crowd_dist(objs, fronts):
    """求拥挤度距离"""
    num_pop, num_dim = objs.shape
    crowd_dist = np.zeros(num_pop)
    for f in fronts:
        # 获取当前前沿面中解的目标值
        objs_f = objs[f, :]
        # 求最大与最小值
        FMax = np.max(objs_f, axis=0)
        FMin = np.min(objs_f, axis=0)
        # 求最大与最小的差，方便归一化
        FRange = FMax - FMin
        FRange[FRange == 0] = np.finfo(np.float32).tiny  # 避免除零

        # 排序索引矩阵
        sorted_indices = np.argsort(objs_f, axis=0)
        f_sorted = np.array(f)[sorted_indices]
        # 设置边界个体的距离为无穷大
        crowd_dist[f_sorted[0, np.arange(num_dim)]] = float('inf')
        crowd_dist[f_sorted[-1, np.arange(num_dim)]] = float('inf')
        # 计算中间个体的拥挤度增量
        dist_increments = (objs_f[sorted_indices[2:], np.arange(num_dim)] - objs_f[
            sorted_indices[:-2], np.arange(num_dim)]) / FRange
        # 累加增量到距离
        np.add.at(crowd_dist, f_sorted[1:-1], dist_increments)

    return crowd_dist


def cal_fitness(ranks, crowd_dist):
    """根据支配前沿数和拥挤度距离计算个体排名(适应度值)"""
    # 初始化排序后的种群索引
    indicator = np.hstack((ranks.reshape(-1, 1), -crowd_dist.reshape(-1, 1)))
    # 使用 np.lexsort 对两列指标进行排序
    indices = np.lexsort((indicator[:, 1], indicator[:, 0]))
    # 排序后的数据
    # sorted_data = indicator[indices]
    # 获取排序下标
    fitness = np.argsort(indices)
    return fitness


def get_uniform_vectors(N, M):
    """
    获取分解后的均匀分布的权重向量
    :param N: 个体数
    :param M: 维度数
    :return: 权重向量
    """
    H1 = 1
    while len(list(itertools.combinations(range(H1 + M), M - 1))) <= N:
        H1 = H1 + 1
    W = np.array(list(itertools.combinations(range(H1 + M - 1), M - 1)))
    W = W - np.repeat(np.array([range(M - 1)]), len(W), axis=0)
    W = (np.hstack((W, np.zeros((len(W), 1)) + H1)) - np.hstack((np.zeros((len(W), 1)), W))) / H1
    return W


def shuffle_matrix_in_row(matrix):
    """使用Knuth-Durstenfeld Shuffle算法对矩阵按行进行打乱"""
    N, D = matrix.shape
    for i in reversed(np.arange(1, D)):
        i_vec = np.zeros((N), dtype=int) + i
        j_vec = np.array(np.random.random(N) * (i + 1), dtype=int)
        matrix[np.arange(N), i_vec], matrix[np.arange(N), j_vec] = matrix[np.arange(N), j_vec], matrix[
            np.arange(N), i_vec]


def shuffle(x):
    """x, random=random.random -> shuffle list x in place; return None.
    Optional arg random is a 0-argument function returning a random
    float in [0.0, 1.0); by default, the standard random.random.
    """
    for i in reversed(np.arange(1, len(x))):
        # pick an element in x[:i+1] with which to exchange x[i]
        j = int(np.random.random() * (i + 1))
        x[i], x[j] = x[j], x[i]
