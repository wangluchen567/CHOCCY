# CHOCCY

## 介绍
基于NumPy构建的优化器框架<br>
CHen's Optimizer Constructed with Core numpY

PS: 框架中包含本人研究工作：<br>
Neural Network-Based Dimensionality Reduction for Large-Scale Binary Optimization with Millions of Variables (NNDREA), IEEE Transactions on Evolutionary Computation <br>
原文下载地址：[IEEE xplore](https://ieeexplore.ieee.org/abstract/document/10530207) / [ResearchGate](https://www.researchgate.net/publication/380393707_Neural_Network-Based_Dimensionality_Reduction_for_Large-Scale_Binary_Optimization_with_Millions_of_Variables#:~:text=In%20this%20paper,%20we%20propose%20a%20dimensionality%20reduction%20method%20to)

## 模型说明
### Algorithms: 优化器的算法框架核心

- Multi: 实现的各种多目标优化算法，包括`NSGA-II`、`MOEA/D`、`NNDREA`等
- Single: 实现的各种单目标优化算法，包括`GA`、`SA`等
- Utility: 实现的各种工具类函数，包括`Crossovers`(交叉函数)、`Mutations`(变异函数)、`Plots`(画图函数)等
- ALGORITHM：所有算法的父类，包含了各种通用函数，所有算法类均继承于该类

### Dataset: 数据集

- Multi: 多目标问题的数据集
- Single: 单目标问题的数据集

### Metrics: 算法的评价指标

- Hypervolume: 计算超体积指标相关函数

### Problems: 问题相关类

- Multi: 多目标相关问题类
- Single: 单目标相关问题类
- PROBLEM: 所有问题的父类，包含通用函数与接口，所有问题类均继承于该类

## 安装教程

1.  最好使用 Anaconda 
2. 必要包: `python>=3.6`、`numpy`、`scipy`

## 使用说明

- `本代码仅供参考学习和学术研究下载`
- `Copy核心代码时请注明出处`

## 效果展示
### 单目标问题

- GA Solve Ackley (num_dec = 1) and GA Solve Ackley (num_dec = 2): <br>
<img src="Results/Ackley1.gif" width="288" height="220"/> <img src="Results/Ackley2.gif" width="288" height="220"/><br/>

- GA Solve Ackley (num_dec = 2)(3D) and GA Solve Ackley (num_dec = 10): <br>
<img src="Results/Ackley3.gif" width="288" height="230"/> <img src="Results/Ackley10.gif" width="288" height="230"/><br/>

- GA Solve TSP (30) and Solve MixFixLabelCluster (120, cluster = 3): <br>
<img src="Results/TSP30.gif" width="288" height="230"/> <img src="Results/MFLC.gif" width="288" height="230"/><br/>

### 多目标问题

- NSGA-II Solve ZDT1 (num_dec = 30) and Solve ZDT3 (num_obj = 30) : <br>
<img src="Results/ZDT1.gif" width="288" height="230"/> <img src="Results/ZDT3.gif" width="288" height="230"/><br/>

- NSGA-II Solve DTLZ1 (num_dec = 7) and Solve DTLZ2 (num_obj = 12) : <br>
<img src="Results/DTLZ1_NSGAII.gif" width="288" height="230"/> <img src="Results/DTLZ2_NSGAII.gif" width="288" height="230"/><br/>

- MOEA/D Solve DTLZ1 (num_dec = 7) and Solve DTLZ2 (num_obj = 12) : <br>
<img src="Results/DTLZ1_MOEAD.gif" width="288" height="230"/> <img src="Results/DTLZ2_MOEAD.gif" width="288" height="230"/><br/>

- Comparison of the performance of NSGA-II and NNDREA for MOKP-10000: <br>
<img src="Results/MOKP10k_NSGAII.gif" width="288" height="230"/> <img src="Results/MOKP10k_NNDREA.gif" width="288" height="230"/><br/>

- Comparison of HV performance between NSGA-II and NNDREA for MOKP-10000: <br>
<img src="Results/MOKP10k_NSGAII.png" width="288" height="230"/> <img src="Results/MOKP10k_NNDREA.png" width="288" height="230"/><br/>

- Comparison of the performance of NSGA-II and NNDREA for MOKP-100000: <br>
<img src="Results/MOKP100k_NSGAII.gif" width="288" height="230"/> <img src="Results/MOKP100k_NNDREA.gif" width="288" height="230"/><br/>

- Comparison of HV performance between NSGA-II and NNDREA for MOKP-100000: <br>
<img src="Results/MOKP100k_NSGAII.png" width="288" height="230"/> <img src="Results/MOKP100k_NNDREA.png" width="288" height="230"/><br/>

(PS: 可以看到即使问题扩大十倍，NNDREA的效果依然不变)
## 参与贡献

Luchen Wang


