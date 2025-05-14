from Problems.Single import TSP
from Algorithms import View, Comparator
from Algorithms.Single import GA, SA, ACO, HGATSP, FI, GFLS
from Datasets.Single.TSPLIB.LoadTSPLIB import load_euc_2d, load_matrix

if __name__ == '__main__':
    # 读取TSPLIB数据集（给定城市坐标位置）
    tsp_data = load_euc_2d("../../Datasets/Single/TSPLIB/eil51.tsp")
    problem = TSP(num_dec=tsp_data['dimension'], data=tsp_data['node_coord'], round_to_int=True)
    # 读取TSPLIB数据集（给定城市坐标的距离矩阵）
    # tsp_data = load_matrix("../../Datasets/Single/TSPLIB/gr120.tsp")
    # problem = TSP(num_dec=tsp_data['dimension'], data=tsp_data['dist_matrix'], is_dist_mat=True, round_to_int=True)
    # 初始化算法字典
    algorithms = dict()
    pop_size, max_iter = 100, 1000
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter)
    algorithms['ACO'] = ACO(pop_size, max_iter)
    algorithms['HGA-TSP'] = HGATSP(pop_size, max_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(max_iter)
    comparator = Comparator(problem, algorithms, show_mode=View.BAR, same_init=True)
    comparator.run()
    comparator.plot(show_mode=View.OBJ)
    algorithms['GFLS'].plot(show_mode=View.PROB)
