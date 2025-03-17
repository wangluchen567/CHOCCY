import warnings
import numpy as np
from scipy.spatial.distance import cdist


def cal_GD(objs, optimum):
    """
    计算代际距离指标(Generational Distance)
    :param objs: 目标值
    :param optimum: 理论最优目标值
    :return: 代际距离指标值
    """
    if optimum is None:
        raise ValueError("optimal targets is None")
    if objs.shape[1] != optimum.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    # 计算给定目标值中每一行与最优目标值中每一行之间的欧式距离
    distance_matrix = cdist(objs, optimum, metric='euclidean')
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_IGD(objs, optimum=None):
    """
    计算逆代际距离指标(Inverted Generational Distance)
    :param objs: 目标值
    :param optimum: 理论最优目标值
    :return: 逆代际距离指标值
    """
    if optimum is None:
        raise ValueError("optimal targets is None")
    if objs.shape[1] != optimum.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    # 计算给定目标值中每一行与最优目标值中每一行之间的欧式距离
    distance_matrix = cdist(objs, optimum, metric='euclidean')
    # 按列取最小值，得到每个最优点到最近点的距离
    min_distances = np.min(distance_matrix, axis=0)
    # 计算最小值的均值
    score = np.mean(min_distances)
    return score


def distance_plus(x, y):
    """
    自定义距离函数：
    计算两个向量之间的
    逐元素差值的最大值的平方和的平方根
    """
    # 取逐元素差值的最大值（与零比较）
    diff = np.maximum(x - y, 0)
    # 计算其平方和的平方根
    return np.sqrt(np.sum(diff ** 2))


def cal_GDPlus(objs, optimum=None):
    """
    计算代际距离+指标(Generational Distance Plus)
    :param objs: 目标值
    :param optimum: 理论最优目标值
    :return: 代际距离+指标值
    """
    if optimum is None:
        raise ValueError("optimal targets is None")
    if objs.shape[1] != optimum.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimum, metric=distance_plus)
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_IGDPlus(objs, optimum=None):
    """
    计算逆代际距离+指标(Inverted Generational Distance Plus)
    :param objs: 目标值
    :param optimum: 理论最优目标值
    :return: 逆代际距离+指标值
    """
    if optimum is None:
        raise ValueError("optimal targets is None")
    if objs.shape[1] != optimum.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimum, metric=distance_plus)
    # 按列取最小值，得到每个最优点到最近点的距离
    distance = np.min(distance_matrix, axis=0)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_HV(objs, optimum=None):
    """
    计算超体积指标(Hyper-volume)
    :param objs: 目标值
    :param optimum: 理论最优目标值
    :return: 超体积指标值
    """
    if objs.ndim == 1:
        objs = objs.reshape(1, -1)
    # 获取目标值维度信息
    N, M = objs.shape
    # 如果未提供理论最优目标值，则参考向量全置1
    if optimum is None:
        refer_array = np.ones(M)
    else:
        refer_array = np.array(optimum)
    # 根据参考点规范化目标值
    f_min = np.min(np.vstack((np.min(objs, axis=0), np.zeros([1, M]))), axis=0)
    f_max = np.max(refer_array, axis=0)
    objs_normalized = (objs - f_min) / np.tile((f_max - f_min) * 1.1, (N, 1))
    objs_normalized = objs_normalized[np.all(objs_normalized <= 1, axis=1)]
    # 参考点设为(1, 1, ...)
    ref_point = np.ones(M)
    # 如果目标值矩阵为空，超体积为0
    if objs_normalized.size == 0:
        score = 0
    elif M < 4:
        # 计算精确的HV值
        pl = np.unique(objs_normalized, axis=0)
        s = [[1, pl]]
        for k in range(M - 1):
            s_ = []
            for i in range(len(s)):
                stemp = Slice(s[i][1], k, ref_point)
                for j in range(len(stemp)):
                    temp = [[stemp[j][0] * s[i][0], np.array(stemp[j][1])]]
                    s_ = Add(temp, s_)
            s = s_
        score = 0
        for i in range(len(s)):
            p = Head(s[i][1])
            score = score + s[i][0] * np.abs(p[-1] - ref_point[-1])
    else:
        # 通过蒙特卡罗方法估计HV值
        sample_num = 1e6
        max_value = ref_point
        min_value = np.min(objs_normalized, axis=0)
        samples = np.random.uniform(np.tile(min_value, (sample_num, 1)), np.tile(max_value, (sample_num, 1)))
        for i in range(len(objs_normalized)):
            domi = np.ones(len(samples), dtype=bool)
            m = 0
            while m <= M - 1 and np.any(domi):
                b = objs_normalized[i][m] <= samples[:, m]
                domi = domi & b
                m += 1
            samples = samples[~domi]
        score = np.prod(max_value - min_value) * (1 - len(samples) / sample_num)

    return score


def Slice(pl: np.ndarray, k: int, ref_point: np.ndarray) -> list:
    p = Head(pl)
    pl = Tail(pl)
    ql = np.array([])
    s = []
    while len(pl):
        ql = Insert(p, k + 1, ql)
        p_ = Head(pl)
        if ql.ndim == 1:
            list_ = [[np.abs(p[k] - p_[k]), np.array([ql])]]
        else:
            list_ = [[np.abs(p[k] - p_[k]), ql]]
        s = Add(list_, s)
        p = p_
        pl = Tail(pl)
    ql = Insert(p, k + 1, ql)
    if ql.ndim == 1:
        list_ = [[np.abs(p[k] - ref_point[k]), np.array([ql])]]
    else:
        list_ = [[np.abs(p[k] - ref_point[k]), ql]]
    s = Add(list_, s)
    return s


def Insert(p: np.ndarray, k: int, pl: np.ndarray) -> np.ndarray:
    flag1 = 0
    flag2 = 0
    ql = np.array([])
    hp = Head(pl)
    while len(pl) and hp[k] < p[k]:
        if len(ql) == 0:
            ql = hp
        else:
            ql = np.vstack((ql, hp))
        pl = Tail(pl)
        hp = Head(pl)
    if len(ql) == 0:
        ql = p
    else:
        ql = np.vstack((ql, p))
    m = max(p.shape)
    while len(pl):
        q = Head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1
        if not (flag1 == 1 and flag2 == 0):
            if len(ql) == 0:
                ql = Head(pl)
            else:
                ql = np.vstack((ql, Head(pl)))
        pl = Tail(pl)
    return ql


def Head(pl: np.ndarray) -> np.ndarray:
    # 取第一行所有元素
    if pl.ndim == 1:
        p = pl
    else:
        p = pl[0]
    return p


def Tail(pl: np.ndarray) -> np.ndarray:
    # 取除去第一行的所有元素
    if pl.ndim == 1 or min(pl.shape) == 1:
        ql = np.array([])
    else:
        ql = pl[1:]
    return ql


def Add(list_: list, s: list) -> list:
    n = len(s)
    m = 0
    for k in range(n):
        if len(list_[0][1]) == len(s[k][1]) and array_equal(list_[0][1], s[k][1]):
            s[k][0] = s[k][0] + list_[0][0]
            m = 1
            break
    if m == 0:
        if n == 0:
            s = list_
        else:
            s.append(list_[0])
    s_ = s
    return s_


try:
    # 尝试导入numba
    from numba import jit


    @jit(nopython=True)
    def array_equal(arr1, arr2, equal_nan=False):
        """
        检查两个数组是否具有相同的形状和元素。
        :param arr1: 第一个输入数组
        :param arr2: 第二个输入数组
        :param equal_nan: 是否将 NaN 视为相等，默认为 False
        :return: 如果两个数组的形状和所有元素都相等，则返回 True，否则返回 False
        """
        # 检查形状是否一致
        if arr1.shape != arr2.shape:
            return False

        # 遍历数组元素进行比较
        for i in range(arr1.size):
            if equal_nan:
                if np.isnan(arr1.flat[i]) and np.isnan(arr2.flat[i]):
                    continue
                elif arr1.flat[i] != arr2.flat[i]:
                    return False
            else:
                if arr1.flat[i] != arr2.flat[i] and not (np.isnan(arr1.flat[i]) and np.isnan(arr2.flat[i])):
                    return False

        return True

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Optimizing problems without using numba acceleration...")
    array_equal = np.array_equal
