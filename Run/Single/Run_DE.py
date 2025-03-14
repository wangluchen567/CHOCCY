from Problems.Single.Ackley import Ackley  # 定义问题后导入问题
from Algorithms.Single.DE import DE  # 导入求解该问题的算法

problem = Ackley(num_dec=2)  # 实例化问题，并指定决策向量大小
# 实例化算法并设置种群大小为100，迭代次数为100，优化过程展示为目标值变化情况
alg = DE(problem, num_pop=100, num_iter=100, show_mode=DE.OAD3)
alg.run()  # 运行算法
# 获取最优解并打印
best, best_obj, best_con = alg.get_best()
print("最优解：", best)
print("最优解的目标值：", best_obj)