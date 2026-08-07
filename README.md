# 🍪 CHOCCY

基于NumPy构建的启发式优化求解器<br>
Chen's Heuristic Optimizer Constructed with Core numpY

## 🔖 项目简介

本项目是一个完全免费且开源的启发式优化求解器，致力于打造一个简单易用、绘图功能丰富、便于算法分析且扩展性强的优化框架。
本项目提供了大量启发式/元启发式算法的实现细节，支持对`实数`、`整数`、`序列`、`(固定)标签`以及`混合`类型问题的优化，
旨在为对优化领域感兴趣的伙伴们提供易于理解的算法学习资源和有力的研究支持💪。

另外，本项目希望为学习、竞赛和科研等领域的伙伴提供高效易用的优化工具。
无论是初学者还是资深研究者，都能借助它快速设计和优化问题，并获得满意结果。
同时，不错的可视化功能可将结果转化为直观图表，方便用于论文撰写和报告制作，助力在学术与实践中取得优异成果🎉。

本项目采用 **木兰宽松许可证第2版（Mulan PSL v2）** 开源，允许学习、竞赛、科研及**商业用途**在内的各种使用场景。

> 📌 **使用说明**：使用本项目时，请保留原始版权声明，并在核心代码复用或引用时注明出处。

## 🌟 主要特性

- 🌍 支持多种`启发式`和`元启发式`优化算法，适用于广泛的问题类型求解
- 🚀 支持对多种问题的优化，包括但不限于`实数`、`整数`、`序列`、`(固定)标签`问题
- 🧩 支持对混合问题的优化，即问题的不同部分可以是不同类型的“混合”问题
- 🛠️ 提供了丰富的性能评估指标和优化算子及函数，可扩展性强
- 📉 支持对多种算法的实时优化比较，并提供丰富的可视化功能
- 📊 支持多种算法对多类问题的优化和求解，并对结果进行比较和可视化
- ⚡ 使用矩阵操作和`numba`的即时编译对核心部分进行优化，显著提高优化速度 

## 🎯 实现算法

- 🧬 遗传(进化)算法
- 🔥 模拟退火算法
- 🔗 差分进化算法
- 🐜 蚁群优化算法
- 🌟 粒子群优化算法
- ⚖️ 多目标优化算法
- 📉 梯度下降优化算法
- 🔍 局部搜索相关算法
- 🚚 路由优化相关算法
- 📝 其他参见[实现清单](https://gitee.com/wang567/CHOCCY/blob/master/docs/IMPLES.md)

## 📚 安装教程

**详细的安装与使用教程请参见[使用指南](https://gitee.com/wang567/CHOCCY/blob/master/docs/GUIDE.md)** 🔥🔥🔥

### 1. 使用 pip 安装（推荐）

直接安装完整版本（包含 numba 加速）：

```bash
pip install choccy
```

如果遇到 numba 相关的报错，或希望安装无 numba 加速的版本：

```bash
pip install choccy[no-numba]
```

如果 tbb 加速不适配当前系统：

```bash
pip install choccy[no-tbb]
```

> 💡 **提示**：以上三种安装方式可根据实际情况选择其一。

### 2. 从 whl 文件安装

从 [Releases 页面](https://gitee.com/wang567/CHOCCY/releases) 下载对应版本的 `.whl` 文件，切换到文件所在目录后执行（以 0.1.1 版本为例）：

```bash
pip install choccy-0.1.1-py3-none-any.whl
```

### 3. 本地源码运行（适用于开发或调试）

如果不希望安装，可以直接下载项目源码进行本地运行和测试。

#### 3.1 使用 Conda 创建虚拟环境（推荐）

建议使用 Anaconda 创建独立的 Python 环境，便于管理依赖包、避免版本冲突。

- 从 [Anaconda 官网](https://www.anaconda.com/download/success) 下载并安装
- 如需特定版本，可访问 [历史版本下载地址](https://repo.anaconda.com/archive/)

创建并激活环境：

```bash
conda create --name choccy_env python=3.9
conda activate choccy_env
```

> 📌 **注意**：本项目支持 Python 3.7 及以上版本，推荐使用 Python 3.9 以获得最佳兼容性。

#### 3.2 安装必需依赖

```bash
pip install numpy scipy matplotlib seaborn tqdm networkx
```

#### 3.3 安装可选加速依赖（建议）

为了获得更快的优化速度，建议安装 numba 和 tbb：

```bash
pip install numba tbb
```

#### 3.4 使用国内镜像源加速下载（可选）

如果下载速度较慢，可尝试使用清华大学镜像源：

```bash
pip install numpy scipy matplotlib seaborn tqdm networkx numba tbb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 📦 项目结构

```
CHOCCY/
├── choccy/                          # 优化求解器核心库（pip 安装内容）
│   ├── algorithms/                  # 算法模块
│   │   ├── multi/                   # 多目标优化算法
│   │   ├── single/                  # 单目标优化算法
│   │   ├── algorithm.py             # 算法基类
│   │   ├── comparator.py            # 单问题多算法比较器
│   │   └── evaluator.py             # 多问题多算法评估器
│   ├── problems/                    # 问题模块
│   │   ├── multi/                   # 多目标优化问题
│   │   ├── single/                  # 单目标优化问题
│   │   └── problem.py               # 问题基类
│   ├── solutions/                   # 解集模块
│   │   └── solutions.py             # 解集基类
│   ├── utilities/                   # 工具模块
│   │   ├── commons/                 # 公共组件
│   │   ├── handler/                 # 数据处理
│   │   ├── logging/                 # 日志处理
│   │   ├── metrics/                 # 性能指标计算
│   │   ├── strategies/              # 策略函数集
│   │   └── visualization/           # 可视化绘图
│   ├── core.py                      # 核心接口
│   └── types.py                     # 枚举类型定义
│
├── demos/                           # 演示程序（需克隆运行）
├── docs/                            # 项目文档
│   └── images/                      # 文档图片资源（动图/截图）
├── examples/                        # 示例代码（需克隆运行）
│   ├── multi/                       # 多目标优化示例
│   ├── single/                      # 单目标优化示例
│   ├── start.py                     # 快速开始示例
│   └── vectorized.py                # 向量化计算示例
├── tests/                           # 单元测试与集成测试
│
├── pyproject.toml                   # 项目配置（构建依赖、工具配置）
└── README.md                        # 项目说明文档（用户入口）
```

> 📌 **目录说明**：
> - `choccy/`：核心库，**会随 `pip install choccy` 一同安装**
> - `examples/`、`demos/`、`tests/`：辅助资源，**仅存在于源码仓库中**（需 `git clone`）
> - `docs/images/`：文档图片资源，用于 README 中的效果展示

> 📚 **支持的算法与问题集**：完整的算法列表和测试问题清单，请参阅 [实现清单](https://gitee.com/wang567/CHOCCY/blob/master/docs/IMPLES.md)

> **✨ PS: 项目中包含本人研究工作** <br>
> Neural Network-Based Dimensionality Reduction for Large-Scale Binary Optimization with Millions of Variables (NNDREA), IEEE Transactions on Evolutionary Computation <br>
> 原文下载地址：[IEEE xplore](https://ieeexplore.ieee.org/abstract/document/10530207) / [ResearchGate](https://www.researchgate.net/publication/380393707_Neural_Network-Based_Dimensionality_Reduction_for_Large-Scale_Binary_Optimization_with_Millions_of_Variables#:~:text=In%20this%20paper,%20we%20propose%20a%20dimensionality%20reduction%20method%20to) (免费下载)


## 🚀 快速开始

### 算法求解自定义问题

#### 基础用法：单个计算

对于简单的小规模问题，可以直接定义单个计算的目标函数：

```python
import numpy as np
from choccy.algorithms.single import DE
from choccy.problems import Problem, create_problem

# 定义目标函数（接收单个解，返回目标值）
def calc_sphere_obj(x):
    return np.sum(x ** 2)

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
```
输出结果（日志部分已省略）：
```
[DE] Iter: 000/100 | Obj: 2.272870e+01 | Feas: 100.0 % | Time: 0.000 s
[DE] Iter: 001/100 | Obj: 2.272870e+01 | Feas: 100.0 % | Time: 0.001 s
...
[DE] Iter: 099/100 | Obj: 1.295769e-20 | Feas: 100.0 % | Time: 0.033 s
[DE] Iter: 100/100 | Obj: 1.295769e-20 | Feas: 100.0 % | Time: 0.033 s

==================================================================
OPTIMIZATION RESULT - DE
==================================================================
Iterations: 100
Runtime: 0.032571 s
Number of Bests: 1
Best Objectives: 1.295769e-20
Best Constraints: N/A
Best Decision Variables:
[1.02604155e-10 4.92958320e-11]
==================================================================
```

#### 进阶用法：向量化计算（推荐）

对于大规模或高维问题，建议使用向量化方式批量计算，能显著提升优化效率：

```python
import numpy as np
from choccy.algorithms.single import DE
from choccy.problems import Problem, create_problem

# 定义目标函数（接收解矩阵，批量返回目标值向量）
def calc_rastrigin_objs(xs):
    # xs 形状为 (n_solutions, n_vars)
    # 返回形状为 (n_solutions,) 或 (n_solutions, 1)
    return np.sum(xs ** 2 - 10 * np.cos(2 * np.pi * xs) + 10, axis=1)

# 创建问题实例
problem = create_problem(
    calc_objs_mat=calc_rastrigin_objs,   # 向量化目标函数（批量计算）
    var_types=Problem.REAL,              # 决策变量类型：实数
    n_vars=2,                            # 决策变量个数
    n_objs=1,                            # 目标个数
    l_bounds=-10,                        # 变量下界
    u_bounds=10                          # 变量上界
)
# 定义并初始化算法
algorithm = DE(n_sols=50, max_iter=100, visual_mode='log')
# 使用算法优化问题
algorithm.optimize(problem)
# 报告优化结果信息
algorithm.report_result()
```
输出结果（日志部分已省略）：
```
[DE] Iter: 000/100 | Obj: 1.729376e+01 | Feas: 100.0 % | Time: 0.000 s
[DE] Iter: 001/100 | Obj: 1.354248e+01 | Feas: 100.0 % | Time: 0.001 s
...
[DE] Iter: 099/100 | Obj: 0.000000e+00 | Feas: 100.0 % | Time: 0.015 s
[DE] Iter: 100/100 | Obj: 0.000000e+00 | Feas: 100.0 % | Time: 0.015 s

==================================================================
OPTIMIZATION RESULT - DE
==================================================================
Iterations: 100
Runtime: 0.015571 s
Number of Bests: 1
Best Objectives: 0.000000e+00
Best Constraints: N/A
Best Decision Variables:
[-4.32080858e-10  1.10147149e-09]
==================================================================
```
> 💡 性能提示：对比两种方式的运行时间（0.033s vs 0.016s），向量化版本的性能优势明显。在处理大规模种群或高维问题时，推荐优先使用 calc_objs_mat 接口。

> 🎨 可视化提示：如需在优化过程中实时绘制动态图像，也推荐使用 calc_objs_mat 接口，可获得更流畅的视觉效果。

#### 可视化示例

以下为使用向量化计算时的优化过程动图（不同参数组合）：

- 左图：`n_vars=2`, `visual_mode='obj'`
- 右图：`n_vars=1`, `visual_mode='h2d'`

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/obj.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/h2d1.gif" height="230"/>
</div>

- 左图：`n_vars=2`, `visual_mode='h2d'`
- 右图：`n_vars=2`, `visual_mode='h3d'`

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/h2d.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/h3d.gif" height="230"/>
</div>

📌 **`visual_mode` 参数说明**：

| 参数      | 含义          | 备注                              |
|---------|-------------|---------------------------------|
| `'obj'` | 目标空间绘制      | 单目标：收敛曲线；多目标：目标空间               |
| `'h2d'` | 决策+目标空间混合绘制 | `n_vars=1`：二维曲线；`n_vars=2`：等高线图 |
| `'h3d'` | 决策+目标空间混合绘制 | 三维曲面图，**要求 `n_vars=2`**         |


## 🌈 效果展示

### 单目标问题优化

> 📌 **说明**：以下动图和图片展示的是项目示例的运行效果。示例代码位于 GitHub 仓库的 `examples/` 目录中，**不会随 `pip install choccy` 一同安装**。如需本地运行，请先克隆项目源码。

### 单目标问题优化

#### 特殊类型问题优化

| 问题类型       | 对应脚本                                                    |
|:-----------|:--------------------------------------------------------|
| TSP（旅行商问题） | `examples/single/optimize/optimize_TSP.py`              |
| 固定标签聚类问题   | `examples/single/optimize/optimize_FixedSizeCluster.py` |

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/tsp.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/cluster.gif" height="230"/>
</div>

#### 多种算法实时对比（相同问题）

| 问题类型            | 对应脚本                                        |
|:----------------|:--------------------------------------------|
| 实数问题（Ackley 函数） | `examples/single/compare/compare_Ackley.py` |
| TSP 问题          | `examples/single/compare/compare_TSP.py`    |

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/compare_ackley.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/compare_tsp.gif" height="230"/>
</div>

#### 多种算法结果对比（静态分析）

| 图表类型 | 说明 |
|:--------|:--------|
| 小提琴图 | 展示目标值分布形态 |
| 核密度估计图 | 展示目标值概率密度 |

**对应脚本**：`examples/single/evaluate/evaluate_Ackley.py`

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/violin.png" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/kde.png" height="230"/>
</div>

> 📌 **注**：为了更清晰地展示对比效果，以上静态图中的迭代次数已缩减为 100 次。

### 多目标问题优化

#### 基准测试问题优化

| 测试问题  | 说明                 | 对应脚本                                       |
|:------|:-------------------|:-------------------------------------------|
| ZDT3  | 双目标优化问题，具有不连续帕累托前沿 | `examples/multi/optimize/optimize_ZDT.py`  |
| DTLZ2 | 多目标优化问题，具有凹球面形帕累托前沿  | `examples/multi/optimize/optimize_DTLZ.py` |

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/zdt3.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/dtlz2.gif" height="230"/>
</div>

#### 多种算法实时对比（多目标背包问题）

对比不同算法在 MOKP（多目标背包问题，1000 维）上的优化表现：

**对应脚本**：`examples/multi/compare/compare_MOKP.py`

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/compare_mokp.gif" height="230"/>
    <img src="https://cdn.jsdelivr.net/gh/wangluchen567/CHOCCY@master/docs/images/mokp_hv.png" height="230"/>
</div>

> 📌 **说明**：右侧图片展示的是 HV（Hypervolume，超体积）指标对比，用于衡量多目标算法的收敛性和多样性。

### 更多示例

完整的示例代码请参阅 Gitee/GitHub 仓库的 [examples/](https://gitee.com/wang567/CHOCCY/tree/master/examples) 目录。


## 📄 数据集优化结果展示

### TSP 数据集优化结果

考虑到部分用户会使用本项目求解旅行商问题（TSP），以下给出 TSPLIB 数据集中部分中小规模实例的优化结果。

| Instance | BKS   | GA    | SA    | ACO   | HGA-TSP | FI    | LS    | GFLS  | Gap(%) |
|----------|-------|-------|-------|-------|---------|-------|-------|-------|--------|
| gr17     | 2085  | 2085  | 2088  | 2085  | 2085    | 2096  | 2088  | 2085  | 0.00   |
| gr24     | 1272  | 1272  | 1279  | 1272  | 1272    | 1361  | 1272  | 1272  | 0.00   |
| eil51    | 426   | 429   | 443   | 438   | 432     | 451   | 458   | 426   | 0.00   |
| eil76    | 538   | 594   | 583   | 565   | 562     | 607   | 600   | 538   | 0.00   |
| berlin52 | 7542  | 7918  | 8034  | 7547  | 7542    | 8118  | 8368  | 7542  | 0.00   |
| KroA100  | 21282 | 26063 | 22238 | 22455 | 21472   | 23373 | 23417 | 21282 | 0.00   |
| KroB100  | 22141 | 26571 | 24078 | 23377 | 22391   | 23958 | 24474 | 22157 | 0.07   |
| KroC100  | 20749 | 25934 | 21991 | 21597 | 22080   | 21817 | 22671 | 20749 | 0.00   |

#### 实验设置

- **种群大小**：`pop_size = 100`
- **迭代次数**：`max_iter = 1000`（保证算法充分收敛）
- **结果取值**：每次优化取多次运行中的**最小值**（由于随机初始化，结果可能略有波动）

> 📌 **补充说明**：上表中 Gap (%) 列展示的是 **GFLS 算法**与 BKS（当前已知的最优解） 的差距。从结果可以看出，GFLS 在多个实例上均能达到或接近已知最优解。

## 🤝 项目贡献

**Maintainer: Luchen Wang**<br>

## ✉️ 联系我们

**邮箱: wangluchen567@qq.com**


