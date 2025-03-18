from Problems.Multi.ZDT.ZDT1 import ZDT1
from Algorithms.Utility.PlotUtils import plot_objs
from Algorithms.Utility.PerfMetrics import cal_GD, cal_GDPlus, cal_IGD, cal_IGDPlus, cal_HV
"""计算评价指标分数测试"""
problem = ZDT1()
pf_points = problem.get_pareto_front(100)
test_objs = pf_points[::10] * 1.1
print("GD:", cal_GD(test_objs, pf_points))
print("GD+:", cal_GDPlus(test_objs, pf_points))
print("IGD:", cal_IGD(test_objs, pf_points))
print("IGD+:", cal_IGDPlus(test_objs, pf_points))
print("HV:", cal_HV(test_objs, pf_points))
plot_objs(test_objs, pareto_front=pf_points)