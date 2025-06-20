from Algorithms import View  # 导入绘图参数类
from Algorithms.Single import DE  # 导入求解问题的算法
from Problems.Single import Ackley  # 定义问题后导入问题
"""快速开始Demo"""

problem = Ackley(num_dec=2)  # 实例化问题，并指定决策向量大小
# 实例化算法并设置种群大小为100，迭代次数为100，优化过程展示为目标值变化情况
algorithm = DE(pop_size=100, max_iter=100, show_mode=View.OBJ)
algorithm.solve(problem)  # 使用该算法求解问题
# 获取最优解并打印
best, best_obj, best_con = algorithm.get_best()
print("最优解：", best)
print("最优解的目标值：", best_obj)
print("算法运行时间(秒)：", algorithm.run_time)