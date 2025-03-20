"""
工具类
Utils
Copyright (c) 2024 LuChen Wang
"""
import time
import warnings
import itertools
import numpy as np

# 设置警告过滤规则，只显示一次
warnings.filterwarnings('once')


def record_time(method):
    """统计运行时间"""

    def timed(*args, **kwargs):
        self = args[0]
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        self.run_time += end_time - start_time
        return result

    return timed


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


def get_dom_between_(objs):
    """得到每对解的支配关系"""
    n = len(objs)
    dom_between = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_dom(objs[i], objs[j]):
                dom_between[i, j] = True
    return dom_between


def fast_nd_sort_(objs):
    """快速非支配排序(非jit加速版)"""
    pop_size = len(objs)  # 获取种群数量
    objs = np.array(objs)  # 将目标转换为numpy数组
    fronts = []  # 初始化各前沿面列表
    ranks = np.zeros(pop_size, dtype=int)  # 每个个体所在的前沿数
    num_dom = np.zeros(pop_size, dtype=int)  # 每一个解被支配的次数初始化为0
    sol_dom = [[] for _ in range(pop_size)]  # 每一个解所支配的解列表

    # 创建比较矩阵以确定支配关系
    for i in range(pop_size):
        # 判断解 i 是否支配其他解
        dominates = np.all(objs[i] <= objs, axis=1) & np.any(objs[i] < objs, axis=1)
        # 记录被解 i 支配的解
        sol_dom[i] = np.where(dominates)[0].tolist()
        # 更新每一个解被支配的次数
        num_dom += dominates.astype(int)

    # 找出第一前沿面
    first_front = np.where(num_dom == 0)[0].tolist()
    ranks[num_dom == 0] = 1  # 将第一前沿面的排序序号设置为1
    fronts.append(first_front)  # 将第一前沿面添加到fronts列表中

    i = 0
    # 迭代处理每一个前沿面
    while i < len(fronts):
        next_front = []  # 初始化下一个前沿面
        for p in fronts[i]:  # 遍历当前前沿面的每一个解
            for q in sol_dom[p]:  # 遍历每一个被当前解支配的解
                num_dom[q] -= 1  # 被支配的次数减1
                if num_dom[q] == 0:  # 如果支配次数减为0，表示该解进入下一前沿面
                    ranks[q] = i + 2  # 排序序号更新
                    next_front.append(q)  # 将解添加到下一前沿面
        if next_front:  # 如果下一个前沿面不为空
            fronts.append(next_front)  # 将下一个前沿面添加到fronts列表中
        i += 1  # 处理下一个前沿面

    return fronts, ranks


def cal_crowd_dist(objs, fronts):
    """求拥挤度距离"""
    pop_size, num_dim = objs.shape
    crowd_dist = np.zeros(pop_size)
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


def cal_ranking(ranks, crowd_dist):
    """根据支配前沿数和拥挤度距离计算个体排名"""
    # 初始化排序后的种群索引
    indicator = np.hstack((ranks.reshape(-1, 1), -crowd_dist.reshape(-1, 1)))
    # 使用 np.lexsort 对两列指标进行排序
    indices = np.lexsort((indicator[:, 1], indicator[:, 0]))
    # 排序后的数据
    # sorted_data = indicator[indices]
    # 获取排序下标
    ranking = np.argsort(indices)
    return ranking


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


def cost_change(dist_mat, n1, n2, n3, n4):
    """计算2-opt的"""
    return dist_mat[n1, n3] + dist_mat[n2, n4] - dist_mat[n1, n2] - dist_mat[n3, n4]


def two_opt_i(tour, dist_mat, i):
    """搜索第i个城市的邻域"""
    improved = False
    idx = np.where(np.array(tour) == i)[0][0]
    tour = np.concatenate((tour[idx:], tour[:idx]))
    for j in range(3, len(tour)):
        n1, n2, n3, n4 = tour[0], tour[1], tour[j - 1], tour[j]
        if cost_change(dist_mat, n1, n2, n3, n4) < -1e-9:
            tour[1:j] = tour[j - 1:0:-1]
            improved = True
            return tour, improved
    return tour, improved


def two_opt_(tour, dist_mat):
    """进行2-opt搜索"""
    improved = False
    for i in range(len(tour)):
        tour, improved = two_opt_i(tour, dist_mat, i)
        if improved:
            return tour, improved
    return tour, improved


try:
    # 尝试导入numba
    from numba import jit, boolean


    @jit(nopython=True)
    def get_dom_between_jit(objs):
        """得到每对解的支配关系"""
        n = len(objs)
        dom_between = np.zeros((n, n), dtype=boolean)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    dom_between[i, j] = True
        return dom_between


    def get_dom_between(objs):
        return get_dom_between_jit(objs)


    @jit(nopython=True)
    def dominates_loop(objs, i):
        n = objs.shape[0]
        m = objs.shape[1]
        dominates = np.zeros(n, dtype=np.bool_)

        for j in range(n):
            all_less_equal = True
            any_less = False

            for k in range(m):
                if objs[i, k] > objs[j, k]:
                    all_less_equal = False
                    break
                elif objs[i, k] < objs[j, k]:
                    any_less = True

            dominates[j] = all_less_equal and any_less

        return dominates


    @jit(nopython=True)
    def fast_nd_sort_jit(objs):
        """快速非支配排序(jit加速版)"""
        pop_size = objs.shape[0]  # 获取种群数量
        fronts = np.zeros((pop_size, pop_size), dtype=np.int16)  # 初始化各前沿面的索引数组
        fronts_trunc = np.zeros(pop_size, dtype=np.int16)  # 各前沿面的索引数组的索引截断
        ranks = np.zeros(pop_size, dtype=np.int16)  # 每个个体所在的前沿数
        num_dom = np.zeros(pop_size, dtype=np.int16)  # 每一个解被支配的次数初始化为0
        sol_dom = np.zeros((pop_size, pop_size), dtype=np.int16)  # 每一个解所支配的解的索引数组
        sol_trunc = np.zeros(pop_size, dtype=np.int16)  # 所支配解的索引截断

        # 创建比较矩阵以确定支配关系
        for i in range(pop_size):
            # 判断解 i 是否支配其他解
            dominates = dominates_loop(objs, i)
            # 得到被解 i 支配的解的索引
            indices = np.where(dominates)[0]
            # 得到索引数组的截断
            sol_trunc[i] = len(indices)
            # 记录被解 i 支配的解的索引
            sol_dom[i][:len(indices)] = indices
            # 更新每一个解被支配的次数
            num_dom += dominates.astype(np.int16)

        # 找出第一前沿面
        first_front = np.where(num_dom == 0)[0]
        ranks[num_dom == 0] = 1  # 将第一前沿面的排序序号设置为1
        fronts[0][:len(first_front)] = first_front  # 将第一前沿面添加到数组中
        front_count = len(first_front)  # 第一前沿面的解数量
        fronts_trunc[0] = front_count  # 记录截断数据

        i = 0
        # 迭代处理每一个前沿面
        while True:
            next_front = np.zeros(pop_size, dtype=np.int16)  # 初始化下一个前沿面
            next_count = 0
            for p in fronts[i][:front_count]:  # 遍历当前前沿面的每一个解
                for q in sol_dom[p][:sol_trunc[p]]:  # 遍历每一个被当前解支配的解
                    num_dom[q] -= 1  # 被支配的次数减1
                    if num_dom[q] == 0:  # 如果支配次数减为0，表示该解进入下一前沿面
                        ranks[q] = i + 2  # 排序序号更新
                        next_front[next_count] = q  # 将解添加到下一前沿面
                        next_count += 1
            if next_count > 0:  # 如果下一个前沿面不为空
                fronts[i + 1][:next_count] = next_front[:next_count]
                front_count = next_count
                fronts_trunc[i + 1] = front_count  # 记录截断数据
                i += 1  # 处理下一个前沿面
            else:  # 若下一个前沿面为空则返回
                break

        return fronts, fronts_trunc, ranks


    def fast_nd_sort(objs):
        fronts_mat, fronts_trunc, ranks = fast_nd_sort_jit(objs)
        fronts_trunc = fronts_trunc[fronts_trunc > 0]
        fronts = [fronts_mat[i][:fronts_trunc[i]].tolist() for i in range(len(fronts_trunc))]
        return fronts, ranks


    @jit(nopython=True)
    def cost_change_jit(dist_mat, n1, n2, n3, n4):
        """计算2-opt的"""
        return dist_mat[n1, n3] + dist_mat[n2, n4] - dist_mat[n1, n2] - dist_mat[n3, n4]


    @jit(nopython=True)
    def two_opt_i_jit(tour, dist_mat, i):
        """搜索第i个城市的邻域"""
        improved = False
        idx = np.where(tour == i)[0][0]
        tour = np.concatenate((tour[idx:], tour[:idx]))
        for j in range(3, len(tour)):
            n1, n2, n3, n4 = tour[0], tour[1], tour[j - 1], tour[j]
            if cost_change_jit(dist_mat, n1, n2, n3, n4) < -1e-9:
                tour[1:j] = tour[j - 1:0:-1]
                improved = True
                return tour, improved
        return tour, improved


    @jit(nopython=True)
    def two_opt_jit(tour, dist_mat):
        """进行2-opt搜索"""
        improved = False
        for i in range(len(tour)):
            tour, improved = two_opt_i_jit(tour, dist_mat, i)
            if improved:
                return tour, improved
        return tour, improved


    def two_opt(tour, dist_mat):
        return two_opt_jit(tour, dist_mat)

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Optimizing problems without using numba acceleration...")
    get_dom_between = get_dom_between_
    fast_nd_sort = fast_nd_sort_
    two_opt = two_opt_
