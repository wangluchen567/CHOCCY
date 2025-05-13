from Problems.Single import TSP
from Algorithms import View, Comparator
from Datasets.Single.TSPLIB.LoadTSPLIB import load_euc_2d
from Algorithms.Single import GA, SA, ACO, HGATSP, FI, GFLS

if __name__ == '__main__':
    # 读取TSPLIB数据集
    tsp_data = load_euc_2d("../../Datasets/Single/TSPLIB/berlin52.tsp")
    problem = TSP(num_dec=tsp_data['dimension'], data=tsp_data['node_coord'], round_to_int=True)
    algorithms = dict()
    pop_size, max_iter = 100, 1000
    algorithms['GA'] = GA(pop_size, max_iter)
    algorithms['SA'] = SA(pop_size, max_iter, init_temp=1e6)
    algorithms['ACO'] = ACO(pop_size, max_iter)
    algorithms['HGA-TSP'] = HGATSP(pop_size, max_iter)
    algorithms['FI'] = FI()
    algorithms['GFLS'] = GFLS(max_iter)
    comparator = Comparator(problem, algorithms, show_mode=View.BAR, same_init=True)
    comparator.run()
    comparator.plot(show_mode=View.OBJ)
    algorithms['GFLS'].plot(show_mode=View.PROB)
