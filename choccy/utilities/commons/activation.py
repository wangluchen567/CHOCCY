# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
激活函数集
"""

import numpy as np


def relu(x):
    return x * (x > 0)


def leaky_relu(x, tiny=0.01):
    return x * (x > 0) + tiny * x * (x <= 0)


def sigmoid(x):
    # 防止指数溢出
    # y = 1 / (1 + exp(-x)), x >= 0
    # y = exp(x) / (1 + exp(x)), x < 0
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def step(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    y[x <= 0] = 0.0
    return y
