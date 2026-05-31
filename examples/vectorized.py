import numpy as np
from choccy.algorithms.single import DE
from choccy.problems import Problem, create_problem


# 定义目标函数（接收解矩阵，批量返回目标值向量）
def calc_rastrigin_objs(xs):
    # xs 形状为 (n_solutions, n_vars)，返回形状可以为 (n_solutions,) 或 (n_solutions, 1)
    return np.sum(xs ** 2 - 10 * np.cos(2 * np.pi * xs) + 10, axis=1)


# 随机种子设置（可选）
np.random.seed(42)
# 创建问题实例
problem = create_problem(
    calc_objs_mat=calc_rastrigin_objs,   # 向量化目标函数（批量计算）
    var_types=Problem.REAL,              # 决策变量类型：实数
    n_vars=2,                            # 决策变量个数
    n_objs=1,                            # 目标个数
    l_bounds=-10,                        # 变量下界
    u_bounds=10                          # 变量上界
)
# 定义并初始化算法
algorithm = DE(n_sols=50, max_iter=100, visual_mode='log')
# 使用算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
