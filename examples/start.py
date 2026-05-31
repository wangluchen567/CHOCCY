import numpy as np
from choccy.algorithms.single import DE
from choccy.problems import Problem, create_problem


# 定义目标函数（接收单个解，返回目标值）
def calc_sphere_obj(x):
    return np.sum(x ** 2)

# 随机种子设置（可选）
np.random.seed(42)
# 创建问题实例
problem = create_problem(
    calc_obj=calc_sphere_obj,     # 目标函数（单个计算）
    var_types=Problem.REAL,       # 决策变量类型：实数
    n_vars=2,                     # 决策变量个数
    n_objs=1,                     # 目标个数
    l_bounds=-100,                # 变量下界
    u_bounds=100                  # 变量上界
)
# 定义并初始化算法
algorithm = DE(n_sols=50, max_iter=100, visual_mode='log')
# 使用算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
