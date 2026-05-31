import numpy as np
from typing import Union, Tuple


def latin_hypercube(low: Union[float, np.ndarray],
                    high: Union[float, np.ndarray],
                    size: Tuple[int, int],
                    centered: bool = False) -> np.ndarray:
    """
    拉丁超立方采样

    :param low: 下界，标量或数组 float or array_like
    :param high: 上界，标量或数组 float or array_like
    :param size: 采样形状 tuple (n_samples, n_dimensions)
    :param centered: True: 取区间中心点, False: 区间内随机点
    :return: 采样结果，形状 (n_samples, n_dimensions)
    """
    # 获取形状
    n_samples, n_dims = size
    # 处理边界广播
    low = np.asarray(low)
    high = np.asarray(high)
    if low.ndim == 0:
        low = np.full(n_dims, low)
    if high.ndim == 0:
        high = np.full(n_dims, high)
    # 创建 (n_samples, n_dims) 形状的排列矩阵
    perms = np.tile(np.arange(1, n_samples + 1).reshape(-1, 1), (1, n_dims))
    # 对每列（每个维度）独立打乱
    for i in range(n_dims):
        perms[:, i] = perms[np.random.permutation(n_samples), i]
    if centered:
        # 中心化版本：取区间中点
        samples_01 = (perms - 0.5) / n_samples
    else:
        # 随机版本：区间内均匀随机
        random_vals = np.random.uniform(0, 1, size=(n_samples, n_dims))
        samples_01 = (perms - random_vals) / n_samples
    # 缩放到目标范围
    return low + samples_01 * (high - low)
