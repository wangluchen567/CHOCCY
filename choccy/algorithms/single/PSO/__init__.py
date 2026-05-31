# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
粒子群优化算法集
Particle Swarm Optimization
"""

from .PSO import PSO
from .BPSO import BPSO
# 提供别名
BinaryPSO = BPSO
ParticleSwarmOptimization = PSO

__all__ = ['PSO', 'ParticleSwarmOptimization', 'BPSO', 'BinaryPSO']