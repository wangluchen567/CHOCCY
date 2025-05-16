"""
计算评价指标工具
Performance Metrics

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
from scipy.spatial.distance import cdist


def cal_gd(objs, optimum):
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
    if len(optimum) == 1:
        warnings.warn("Only one theoretical optimal solution has been provided, "
                      "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的欧式距离
    distance_matrix = cdist(objs, optimum, metric='euclidean')
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_igd(objs, optimum=None):
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
    if len(optimum) == 1:
        warnings.warn("Only one theoretical optimal solution has been provided, "
                      "which may be a reference point. Please use HV to calculate the score")
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


def cal_gd_plus(objs, optimum=None):
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
    if len(optimum) == 1:
        warnings.warn("Only one theoretical optimal solution has been provided, "
                      "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimum, metric=distance_plus)
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_igd_plus(objs, optimum=None):
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
    if len(optimum) == 1:
        warnings.warn("Only one theoretical optimal solution has been provided, "
                      "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimum, metric=distance_plus)
    # 按列取最小值，得到每个最优点到最近点的距离
    distance = np.min(distance_matrix, axis=0)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def cal_hv(objs, optimum=None):
    """
    计算超体积指标(Hyper-volume)
    :param objs: 目标值
    :param optimum: 理论最优目标值或参考点
    :return: 超体积指标值
    """
    if objs.ndim == 1:
        objs = objs.reshape(1, -1)
    # 获取目标值维度信息
    obj_size, obj_dim = objs.shape
    # 如果未提供理论最优目标值，则参考点向量全置1
    if optimum is None:
        refer_array = np.ones(obj_dim)
    else:
        refer_array = np.array(optimum)
    # 根据参考点向量规范化目标值
    f_min = np.min(np.vstack((np.min(objs, axis=0), np.zeros([1, obj_dim]))), axis=0)
    f_max = np.max(refer_array, axis=0)
    objs_norm = (objs - f_min) / np.tile((f_max - f_min) * 1.1, (obj_size, 1))
    objs_norm = objs_norm[np.all(objs_norm <= 1, axis=1)]
    # 参考点向量设置为全1向量
    ref_point = np.ones(obj_dim)
    # 如果目标值矩阵为空，超体积为0
    if objs_norm.size == 0:
        score = 0
    elif obj_dim < 4:
        # 计算精确的HV值
        score = cal_hv_accurate(objs_norm, ref_point)
    else:
        # 通过蒙特卡罗方法估计HV值
        score = cal_hv_estimated(objs_norm, ref_point)
    return score


def cal_hv_estimated_(objs_norm, ref_point, sample_num=1e6):
    """
    使用蒙特卡罗方法估计HV值

    Code References:
        PlatEMO(https://github.com/BIMK/PlatEMO)
    :param objs_norm: 规范化后的目标向量
    :param ref_point: 参考点向量
    :param sample_num: 采样点个数
    :return:  HV分数值
    """
    sample_num = int(sample_num)
    max_value = ref_point
    min_value = np.min(objs_norm, axis=0)
    samples = np.random.uniform(np.tile(min_value, (sample_num, 1)), np.tile(max_value, (sample_num, 1)))
    for i in range(len(objs_norm)):
        domi = np.ones(len(samples), dtype=bool)
        m = 0
        while m <= objs_norm.shape[1] - 1 and np.any(domi):
            b = objs_norm[i][m] <= samples[:, m]
            domi = domi & b
            m += 1
        samples = samples[~domi]
    score = np.prod(max_value - min_value) * (1 - len(samples) / sample_num)
    return score


def cal_hv_accurate_(objs_norm, ref_point):
    """
    计算精确的HV值

    References:
        A Faster Algorithm for Calculating Hypervolume,
        Lyndon While, Philip Hingston, Luigi Barone, and Simon Huband
    Code References:
        PlatEMO(https://github.com/BIMK/PlatEMO)
    :param objs_norm: 规范化后的目标向量
    :param ref_point: 参考点向量
    :return: HV分数值
    """
    pl = np.unique(objs_norm, axis=0)
    s = [[1, pl]]
    for k in range(objs_norm.shape[1] - 1):
        s_ = []
        for i in range(len(s)):
            stemp = _slice(s[i][1], k, ref_point)
            for j in range(len(stemp)):
                temp = [[stemp[j][0] * s[i][0], np.array(stemp[j][1])]]
                s_ = _add(temp, s_)
        s = s_
    score = 0
    for i in range(len(s)):
        p = _head(s[i][1])
        score = score + s[i][0] * np.abs(p[-1] - ref_point[-1])
    return score


def _slice(pl: np.ndarray, k: int, ref_point: np.ndarray) -> list:
    p = _head(pl)
    pl = _tail(pl)
    ql = np.array([])
    s = []
    while len(pl):
        ql = _insert(p, k + 1, ql)
        p_ = _head(pl)
        if ql.ndim == 1:
            list_ = [[np.abs(p[k] - p_[k]), np.array([ql])]]
        else:
            list_ = [[np.abs(p[k] - p_[k]), ql]]
        s = _add(list_, s)
        p = p_
        pl = _tail(pl)
    ql = _insert(p, k + 1, ql)
    if ql.ndim == 1:
        list_ = [[np.abs(p[k] - ref_point[k]), np.array([ql])]]
    else:
        list_ = [[np.abs(p[k] - ref_point[k]), ql]]
    s = _add(list_, s)
    return s


def _insert(p: np.ndarray, k: int, pl: np.ndarray) -> np.ndarray:
    flag1 = 0
    flag2 = 0
    ql = np.array([])
    hp = _head(pl)
    while len(pl) and hp[k] < p[k]:
        if len(ql) == 0:
            ql = hp
        else:
            ql = np.vstack((ql, hp))
        pl = _tail(pl)
        hp = _head(pl)
    if len(ql) == 0:
        ql = p
    else:
        ql = np.vstack((ql, p))
    m = max(p.shape)
    while len(pl):
        q = _head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1
        if not (flag1 == 1 and flag2 == 0):
            if len(ql) == 0:
                ql = _head(pl)
            else:
                ql = np.vstack((ql, _head(pl)))
        pl = _tail(pl)
    return ql


def _head(pl: np.ndarray) -> np.ndarray:
    # 取第一行所有元素
    if pl.ndim == 1:
        p = pl
    else:
        p = pl[0]
    return p


def _tail(pl: np.ndarray) -> np.ndarray:
    # 取除去第一行的所有元素
    if pl.ndim == 1 or min(pl.shape) == 1:
        ql = np.array([])
    else:
        ql = pl[1:]
    return ql


def _add(list_: list, s: list) -> list:
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
    import numba as nb
    from numba import jit
    from numba.typed import List


    @jit(nopython=True, cache=True)
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


    @jit(nopython=True, cache=True)
    def _head_jit(pl):
        if pl.shape[0] == 0:
            return np.empty(0)
        return pl[0]


    @jit(nopython=True, cache=True)
    def _tail_jit(pl):
        if pl.shape[0] <= 1:
            return np.empty((0, pl.shape[1]))
        return pl[1:]


    @jit(nopython=True, cache=True)
    def _add_jit(list_, s):
        num_s = len(s)
        found = False
        new_factor, new_ql = list_[0]
        for i in range(num_s):
            old_factor, old_ql = s[i]
            if array_equal(new_ql, old_ql):
                s[i] = (old_factor + new_factor, old_ql)
                found = True
                break
        if not found:
            if num_s == 0:
                return list_
            else:
                s.append((new_factor, new_ql))
        return s


    @jit(nopython=True, cache=True)
    def _insert_jit(p, k, pl):
        m = p.shape[0]
        ql = np.zeros((0, m), dtype=p.dtype)
        i = 0
        while i < pl.shape[0] and pl[i, k] < p[k]:
            ql = np.vstack((ql, pl[i:i + 1]))
            i += 1
        ql = np.vstack((ql, p.reshape(1, -1)))
        while i < pl.shape[0]:
            q = pl[i]
            dominated = True
            has_less = False
            for j in range(k, m):
                if p[j] > q[j]:
                    dominated = False
                    break
                elif p[j] < q[j]:
                    has_less = True
            if not (dominated and has_less):
                ql = np.vstack((ql, q.reshape(1, -1)))
            i += 1
        return ql


    @jit(nopython=True, cache=True)
    def _slice_jit(pl, k, ref_point):
        s = List()
        if pl.shape[0] == 0:
            s.append((1.0, pl))
            return s
        p = _head_jit(pl)
        pl = _tail_jit(pl)
        ql = np.zeros((0, pl.shape[1]), dtype=pl.dtype)
        while pl.shape[0] > 0:
            ql = _insert_jit(p, k + 1, ql)
            p_next = _head_jit(pl)
            list_ = List()
            delta = np.abs(p[k] - p_next[k])
            list_.append((delta, ql))
            s = _add_jit(list_, s)
            p = p_next
            pl = _tail_jit(pl)
        ql = _insert_jit(p, k + 1, ql)
        list_ = List()
        delta = np.abs(p[k] - ref_point[k])
        list_.append((delta, ql))
        s = _add_jit(list_, s)
        return s


    def cal_hv_accurate(objs_norm, ref_point):
        """
        精确计算HV值(使用numba加速)

        References:
            A Faster Algorithm for Calculating Hypervolume,
            Lyndon While, Philip Hingston, Luigi Barone, and Simon Huband
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param objs_norm: 规范化后的目标向量
        :param ref_point: 参考点向量
        :return: HV分数值
        """
        # 删除重复项 Remove duplicates
        objs_norm = np.unique(objs_norm, axis=0)
        if objs_norm.size == 0:
            return 0.0
        return _cal_hv_accurate_jit(objs_norm, ref_point)


    @jit(nopython=True, cache=True)
    def _cal_hv_accurate_jit(objs_norm, ref_point):
        pl = objs_norm
        s = List()
        # Initialize with explicit typing
        s.append((1.0, pl))
        for k in range(objs_norm.shape[1] - 1):
            s_new = List()
            # 指定元素后才能确定类型
            s_new.append((1.0, np.empty((0, pl.shape[1]))))
            for i in range(len(s)):
                factor, points = s[i]
                if points.shape[0] == 0:
                    continue
                slice_temp = _slice_jit(points, k, ref_point)
                for item in slice_temp:
                    delta, ql = item
                    if ql.shape[0] == 0:
                        continue
                    new_factor = factor * delta
                    temp = List()
                    temp.append((new_factor, ql))
                    s_new = _add_jit(temp, s_new)
            s = s_new
        score = 0.0
        for i in range(len(s)):
            factor, points = s[i]
            if points.shape[0] == 0:
                continue
            p = _head_jit(points)
            score += factor * np.abs(p[-1] - ref_point[-1])
        return score

    @jit(nopython=True, parallel=True, cache=True)
    def cal_hv_estimated(objs_norm, ref_point, sample_num=1e6):
        """
        使用蒙特卡罗方法估计HV值(numba多核并行加速)

        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param objs_norm: 规范化后的目标向量
        :param ref_point: 参考点向量
        :param sample_num: 采样点个数
        :return:  HV分数值
        """
        sample_num = int(sample_num)
        dim = objs_norm.shape[1]
        n_points = objs_norm.shape[0]

        # 计算最小边界
        min_values = np.empty(dim)
        for d in range(dim):
            min_val = np.inf
            for i in range(n_points):
                if objs_norm[i, d] < min_val:
                    min_val = objs_norm[i, d]
            min_values[d] = min_val

        # 生成蒙特卡洛样本
        samples = np.empty((sample_num, dim))
        for i in nb.prange(sample_num):
            for d in range(dim):
                samples[i, d] = np.random.uniform(min_values[d], ref_point[d])

        # 初始化活跃样本标记
        active = np.ones(sample_num, dtype=nb.bool_)

        # 并行处理每个样本
        for i in range(n_points):
            current_obj = objs_norm[i]
            for j in nb.prange(sample_num):
                if active[j]:
                    dominated = True
                    # 检查是否被当前目标点支配
                    for m in range(dim):
                        if samples[j, m] < current_obj[m]:
                            dominated = False
                            break
                    if dominated:
                        active[j] = False

        # 计算最终结果
        valid_count = sample_num - np.sum(active)
        volume = 1.0
        for d in range(dim):
            volume *= (ref_point[d] - min_values[d])

        return volume * valid_count / sample_num

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Optimizing problems without using numba acceleration...")
    array_equal = np.array_equal
    cal_hv_accurate = cal_hv_accurate_
    cal_hv_estimated = cal_hv_estimated_
