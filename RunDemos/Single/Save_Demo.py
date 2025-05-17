from Algorithms import View  # 导入绘图参数类
from Algorithms.Single import DE  # 导入求解问题的算法
from Problems.Single import Ackley  # 定义问题后导入问题
"""保存文件Demo"""

problem = Ackley(num_dec=2)  # 实例化问题，并指定决策向量大小
# 实例化算法并设置种群大小为100，迭代次数为100，优化过程输出为日志
algorithm = DE(pop_size=100, max_iter=100, show_mode=View.LOG)
# 开启日志保存功能（日志不仅会输出到控制台，还会保存到日志文件中）
# algorithm.enable_file_logging()  # 这里可以给定日志文件名称参数
algorithm.solve(problem)  # 使用该算法求解问题
# 获取最优解并打印
best, best_obj, best_con = algorithm.get_best()
print("最优解：", best)
print("最优解的目标值：", best_obj)
print("算法运行时间(秒)：", algorithm.run_time)
# 保存得到的最优解和种群
algorithm.save_best()  # 保存优化求解后得到的最优解相关信息
algorithm.save_pop()  # 保存优化求解后得到的整个种群相关信息
algorithm.save_history()  # 保存优化求解后得到的种群所有历史相关信息
