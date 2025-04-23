from Problems.Multi import ZDT1
from Algorithms.Utility.PlotUtils import plot_objs
from Algorithms.Utility.PerfMetrics import cal_gd, cal_gd_plus, cal_igd, cal_igd_plus, cal_hv
"""计算评价指标分数测试"""
problem = ZDT1()
pf_points = problem.get_pareto_front(100)
test_objs = pf_points[::10] * 1.1
print("GD:", cal_gd(test_objs, pf_points))
print("GD+:", cal_gd_plus(test_objs, pf_points))
print("IGD:", cal_igd(test_objs, pf_points))
print("IGD+:", cal_igd_plus(test_objs, pf_points))
print("HV:", cal_hv(test_objs, pf_points))
plot_objs(test_objs, pareto_front=pf_points)