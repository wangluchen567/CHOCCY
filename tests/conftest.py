"""
全局配置：设置随机种子确保测试可复现
"""
import numpy as np


def seed_everything(seed=42):
    np.random.seed(seed)
