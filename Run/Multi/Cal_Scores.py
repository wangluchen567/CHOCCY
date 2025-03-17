from Problems.Multi.ZDT.ZDT1 import ZDT1
from Algorithms.Utility.PerfMetrics import cal_GD, cal_GDPlus, cal_IGD, cal_IGDPlus, cal_HV
"""计算评价指标分数测试"""
problem = ZDT1()
optimum = problem.get_pareto_front(100)
test_data = optimum[::10] * 1.1
print("GD:", cal_GD(test_data, optimum))
print("GD+:", cal_GDPlus(test_data, optimum))
print("IGD:", cal_IGD(test_data, optimum))
print("IGD+:", cal_IGDPlus(test_data, optimum))
print("HV:", cal_HV(test_data, optimum))