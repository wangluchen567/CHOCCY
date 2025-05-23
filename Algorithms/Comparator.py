"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Union
from Algorithms import View
from Algorithms import ALGORITHM
from Problems.PROBLEM import PROBLEM
from mpl_toolkits.mplot3d import Axes3D


class Comparator(ALGORITHM):

    def __init__(self,
                 problem: PROBLEM,
                 algorithms: Union[list, dict],
                 pop_size: Union[int, None] = None,
                 max_iter: Union[int, None] = None,
                 same_init: bool = False,
                 show_colors: Union[list, None] = None,
                 show_mode: Union[int, View] = 0):
        """
        算法比较器(用于对比多个算法效果)
        :param problem: 问题对象
        :param algorithms: 需要对比的算法集合(字典或列表)
        :param pop_size: 每种算法初始化的种群大小
        :param max_iter: 每种算法的迭代次数
        :param same_init: 所有算法是否初始化相同
        :param show_colors: 指定每种算法的展示颜色(名称或HEX)
        :param show_mode: 绘图模式(与ALGORITHM相同)
        """
        super().__init__(pop_size, max_iter)
        self.problem = problem
        self.max_iter_ = max_iter
        self.same_init = same_init
        self.show_mode = show_mode
        self.algorithms = self._format(algorithms)
        self.colors = self.get_colors() if show_colors is None else show_colors
        # 初始化评价指标类型(单目标为适应度, 多目标默认为超体积指标)
        self.score_type = 'Fitness' if self.problem.num_obj == 1 else 'HV'
        # 若没有指定种群大小则默认使用算法集合中第一个算法的种群大小(可能会有bug)
        self.pop_size = next(iter(self.algorithms.values())).pop_size if self.pop_size is None else self.pop_size

    @staticmethod
    def _format(value):
        """格式化列表(将列表变为字典)"""
        if isinstance(value, dict):
            return value
        elif isinstance(value, list):
            return {type(item).__name__: item for item in value}
        else:
            raise ValueError("This type of conversion is not supported")

    def init_comparator(self):
        """初始化比较器"""
        max_iter = 0  # 统计最大迭代次数
        self.init_algorithm(self.problem)
        # 若使用相同初始化则先初始化种群
        pop = self.init_pop() if self.same_init else None
        # 初始化所有算法
        for alg in self.algorithms.values():
            # 对比算法时各个算法的进度条关闭
            alg.show_mode = -1
            # 初始化算法
            if self.same_init:
                # 若使用相同初始化
                alg.init_algorithm(self.problem, pop.copy())
            else:
                alg.init_algorithm(self.problem)
            # 检查是否有单独运行一步
            if 'run_step' not in type(alg).__dict__:
                # 如果算法没有覆写单独运行一步，则全部运行
                alg.show_mode = 0  # 若全部运行则展示bar
                alg.run()
            else:
                # 得到最大迭代次数
                max_iter = max(max_iter, alg.max_iter)
            if self.show_mode == self.SCORE:
                # 若展示指标值则需每步计算指标值
                alg.record_score()
        # 迭代次数若为空则根据给定算法最大迭代次数(max_iter_防止初始化干扰)
        self.max_iter = max_iter if self.max_iter_ is None else self.max_iter_

    def run(self):
        """运行多个算法的实时比较"""
        # 初始化比较器（所有算法）
        self.init_comparator()
        # 绘制初始状态图
        self.plot(n_iter=0, pause=True)
        for i in self.get_iterator():
            for alg in self.algorithms.values():
                if i < alg.max_iter:
                    # 若算法未结束迭代
                    alg.run_step(i)
                else:
                    # 若算法已结束迭代
                    alg.record()
                if self.show_mode == self.SCORE:
                    # 若展示指标值则需每步计算指标值
                    alg.record_score()
            # 绘制迭代过程中每步状态
            self.plot(n_iter=i + 1, pause=True)
        self.prints()

    def set_score_type(self, score_type):
        """重新设置指标分数类型(多目标)"""
        if self.problem.num_obj == 1:
            warnings.warn("Single objective problem cannot set score type")
            return
        if score_type not in self.score_types:
            raise ValueError(f"There is no {score_type} score type")
        self.score_type = score_type
        # 将所有算法重设指标分数类型
        for alg in self.algorithms.values():
            alg.set_score_type(score_type)

    def prints(self, dec=6, show_con=False):
        """
        格式化打印多个算法的对比结果
        :param dec: 精确到小数点后的位数
        :param show_con: 是否展示约束结果
        """
        if self.problem.num_obj == 1:
            self.print_single(dec, show_con)
        else:
            self.print_multi(dec)

    def print_single(self, dec=6, show_cons=False):
        """格式化打印多个算法的对比结果(单目标)"""
        problem_name = type(self.problem).__name__
        titles = ["Algorithm"]
        objs = [problem_name]
        cons = [" "]
        times = ["time(s)"]
        col_widths = [max(len("Algorithm"), len(problem_name)) + 3]
        if show_cons:
            titles.append("type")
            objs.append("obj_value")
            cons.append("con_value")
            col_widths.append(len(objs[-1]) + 3)
        for (name, alg) in self.algorithms.items():
            titles.append(name)
            objs.append(f"{alg.best_obj[0]:.{dec}e}")
            times.append(f"{alg.run_time:.{dec}e}")
            if show_cons:
                cons.append(f"{alg.best_con[0]:.{dec}e}")
            col_widths.append(max(len(titles[-1]), len(objs[-1])) + 3)
        titles_format = " ".join(f"{t:<{w}}" for t, w in zip(titles, col_widths))
        objs_format = " ".join(f"{v:<{w}}" for v, w in zip(objs, col_widths))
        cons_format = ""
        if show_cons:
            cons_format = " ".join(f"{c:<{w}}" for c, w in zip(cons, col_widths))
        times_format = " ".join(f"{t:<{w}}" for t, w in zip(times, col_widths))
        print(titles_format)
        print(objs_format)
        print(times_format)
        print(cons_format)

    def print_multi(self, dec=6):
        """格式化打印多个算法的对比结果(多目标)"""
        problem_name = type(self.problem).__name__
        titles = ["Algorithm"]
        scores = [problem_name]
        times = ["time(s)"]
        col_widths = [max(len("Algorithm"), len(problem_name)) + 3]
        for (name, alg) in self.algorithms.items():
            titles.append(name)
            metric_value = alg.cal_score(best_obj=alg.best_obj)
            scores.append(f"{metric_value:.{dec}e}")
            times.append(f"{alg.run_time:.{dec}e}")
            col_widths.append(max(len(titles[-1]), len(scores[-1])) + 3)
        titles_format = " ".join(f"{t:<{w}}" for t, w in zip(titles, col_widths))
        scores_format = " ".join(f"{s:<{w}}" for s, w in zip(scores, col_widths))
        times_format = " ".join(f"{t:<{w}}" for t, w in zip(times, col_widths))
        print(titles_format)
        print(scores_format)
        print(times_format)

    def get_colors(self):
        """绘图颜色设置"""
        num_colors = len(self.algorithms)
        if num_colors <= 3:
            # 若数量少则直接指定颜色
            colors = ['blue', 'red', 'green']
            return colors
        # 否则从彩虹色图中采样颜色
        # 检查 Matplotlib 版本
        if matplotlib.__version__ >= '3.7':
            rainbow_cmap = plt.colormaps['rainbow']
        else:
            rainbow_cmap = plt.cm.get_cmap('rainbow')
        # 生成 num_colors 种颜色
        raw_colors = rainbow_cmap(np.linspace(0, 1, num_colors))
        # 将颜色转换为十六进制格式
        colors = [mcolors.to_hex(c) for c in raw_colors]
        return colors

    def plot(self, show_mode=None, n_iter=None, pause=False, sym=True):
        """
        绘图函数，根据不同模式进行绘图
        :param show_mode: 绘图模式，参见View类
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param sym: 是否进行完全对称图像的绘制（只对MIX绘图有效）
        """
        if n_iter == self.max_iter:
            # 最后一次迭代不再使用停顿展示
            pause = False
        if show_mode is not None:
            self.show_mode = show_mode
        if show_mode == self.SCORE:
            # 若单独指定展示指标分数，则检查并计算评价指标
            for alg in self.algorithms.values():
                if len(alg.scores) == 0:
                    alg.get_scores()
        if self.show_mode == self.NONE or self.show_mode == self.BAR:
            pass
        elif self.show_mode == self.OBJ:
            self.plot_objs(n_iter, pause)
        elif self.show_mode == self.DEC:
            self.plot_decs(n_iter, pause)
        elif self.show_mode == self.MIX2D:
            self.plot_objs_decs(n_iter, pause, sym=sym)
        elif self.show_mode == self.MIX3D:
            self.plot_objs_decs(n_iter, pause, sym=sym, contour=False)
        elif self.show_mode == self.SCORE:
            self.plot_scores(n_iter, pause)
        else:
            raise ValueError(f"There is no such plotting mode: {self.show_mode}")

    def plot_decs(self, n_iter=None, pause=False, pause_time=0.06):
        """
        绘制种群个体决策向量
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        """
        plt.clf()
        if self.problem.num_dec == 1:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                X_stack = np.hstack((pop, pop))
                plt.plot(np.arange(0, 2), X_stack[0, :], c=self.colors[idx], alpha=0.5, label=name)
                for i in range(1, len(X_stack)):
                    plt.plot(np.arange(0, 2), X_stack[i, :], c=self.colors[idx], alpha=0.5)
            plt.xlim((0, 1))
            plt.xlabel('dim')
            plt.ylabel('x')
        elif self.problem.num_dec == 2:
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                plt.scatter(pop[:, 0], pop[:, 1], marker="o", c=self.colors[idx], label=name)
            plt.xlabel('x')
            plt.ylabel('y')
        elif self.problem.num_dec == 3:
            ax = plt.subplot(111, projection='3d')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
            for idx, name, alg in enumerate(self.algorithms.items()):
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                ax.scatter(pop[:, 0], pop[:, 1], pop[:, 2],
                           marker="o", c=self.colors[idx], label=name)
            # 设置三维图像角度(仰角方位角)
            # ax.view_init(elev=30, azim=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                plt.plot(np.arange(1, self.problem.num_dec + 1), pop[0, :],
                         c=self.colors[idx], alpha=0.5, label=name)
                for i in range(1, len(pop)):
                    plt.plot(np.arange(1, self.problem.num_dec + 1), pop[i, :],
                             c=self.colors[idx], alpha=0.5)
            plt.xlabel('dim')
            plt.ylabel('x')
        plt.grid(True)
        plt.legend()
        if n_iter is None:
            plt.title("Decisions")
        else:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()

    def plot_objs(self, n_iter=None, pause=False, pause_time=0.06):
        """
        绘制种群目标值
        :param n_iter: 迭代次数，绘制指定迭代次数下的图像
        :param pause: 是否短暂暂停显示（不建议外部调用）
        :param pause_time: 短暂暂停时长（不建议外部调用）
        """
        plt.clf()
        if self.problem.num_obj == 1:
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                objs = alg.objs_history if pause or n_iter is None else alg.objs_history[:n_iter + 1]
                x = np.arange(len(objs))
                objs_ = np.concatenate(objs, axis=1).T
                objs_min = np.min(objs_, axis=1)
                objs_max = np.max(objs_, axis=1)
                # 填充最小值和最大值之间的区域
                plt.fill_between(x, objs_min, objs_max, color=self.colors[idx], alpha=0.2)
                plt.plot(x, objs_min, marker=".", c=self.colors[idx], label=name, alpha=0.6)
            plt.xlabel('n_iter')
            plt.ylabel('fitness')
        elif self.problem.num_obj == 2:
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                objs = alg.objs if pause or n_iter is None else alg.objs_history[n_iter]
                plt.scatter(objs[:, 0], objs[:, 1], marker="o", c=self.colors[idx], label=name, alpha=0.6)
            # 绘制最优前沿面
            if self.problem.pareto_front is not None:
                plt.plot(self.problem.pareto_front[:, 0], self.problem.pareto_front[:, 1], marker="", c='gray')
            plt.xlabel('obj1')
            plt.ylabel('obj2')
        elif self.problem.num_obj == 3:
            ax = plt.subplot(111, projection='3d')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                objs = alg.objs if pause or n_iter is None else alg.objs_history[n_iter]
                ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2],
                           marker="o", c=self.colors[idx], label=name, alpha=0.6)
            # 绘制最优前沿面
            if self.problem.pareto_front is not None:
                x = self.problem.pareto_front[:, 0]
                y = self.problem.pareto_front[:, 1]
                z = self.problem.pareto_front[:, 2]
                # 绘制三维三角曲面（面透明且边浅灰）（支持不规则图形）
                ax.plot_trisurf(x, y, z, edgecolor='gray', color=(1, 1, 1, 0), linewidth=0.16)
            # 设置三维图像角度(仰角方位角)
            ax.view_init(elev=30, azim=30)
            ax.set_xlabel('obj1')
            ax.set_ylabel('obj2')
            ax.set_zlabel('obj3')
        else:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                objs = alg.objs if pause or n_iter is None else alg.objs_history[n_iter]
                plt.plot(np.arange(1, self.problem.num_obj + 1), objs[0, :],
                         c=self.colors[idx], alpha=0.6, label=name)
                for i in range(1, len(objs)):
                    plt.plot(np.arange(1, self.problem.num_obj + 1), objs[i, :],
                             c=self.colors[idx], alpha=0.6)
            plt.xlabel('dim')
            plt.ylabel('obj')
        plt.grid(True)
        plt.legend()
        if n_iter is None:
            plt.title("Objectives")
        else:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()

    def plot_objs_decs(self, n_iter=None, pause=False, pause_time=0.06, contour=True, sym=True):
        """在特定条件下可同时绘制决策向量与目标值"""
        plt.clf()
        if self.problem.num_obj > 1:
            raise ValueError("The objective vector dimension must be 1 dimension")
        if self.problem.num_dec == 1:
            all_pop = np.empty(shape=(0, 1))
            for idx, (name, alg) in enumerate(self.algorithms.items()):
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                objs = alg.objs if pause or n_iter is None else alg.objs_history[n_iter]
                plt.scatter(pop, objs, marker="o", c=self.colors[idx], label=name, alpha=0.6)
                all_pop = np.concatenate((all_pop, pop))
            # 对问题进行采样绘制问题图像
            if sym is True:  # 对称图像绘制
                x_min, x_max = -np.max(np.abs(all_pop)), np.max(np.abs(all_pop))
            else:
                x_min, x_max = np.min(all_pop), np.max(all_pop)
            x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
            y = self.problem.cal_objs(x)
            # 设置x轴和y轴使用科学计数法
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            plt.plot(x, y)
            plt.xlabel('x')
            plt.ylabel('obj')
        elif self.problem.num_dec == 2:
            # 得到所有算法的种群状态
            all_pop = np.empty(shape=(0, 2))
            for alg in self.algorithms.values():
                pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                all_pop = np.concatenate((all_pop, pop))
            # 对问题进行采样绘制问题图像
            if sym is True:  # 对称图像绘制
                x_min, x_max = -np.max(np.abs(all_pop)), np.max(np.abs(all_pop))
                y_min, y_max = -np.max(np.abs(all_pop)), np.max(np.abs(all_pop))
            else:
                x_min, x_max = np.min(all_pop[:, 0]), np.max(all_pop[:, 0])
                y_min, y_max = np.min(all_pop[:, 1]), np.max(all_pop[:, 1])
            x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
            y = np.linspace(y_min, y_max, 1000).reshape(-1, 1)
            X, Y = np.meshgrid(x, y)
            decs_pack = np.concatenate((np.expand_dims(X, -1), np.expand_dims(Y, -1)), -1).reshape(-1, 2)
            objs_pack = self.problem.cal_objs(decs_pack)
            Z = objs_pack.reshape(X.shape)
            if contour:
                # 设置x轴和y轴使用科学计数法
                plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
                contour_ = plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 20))
                plt.clabel(contour_, inline=True, fontsize=8)  # 添加等高线标签
                for idx, (name, alg) in enumerate(self.algorithms.items()):
                    pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                    plt.scatter(pop[:, 0], pop[:, 1], marker="o", c=self.colors[idx], label=name, alpha=0.6)
                plt.xlabel('x')
                plt.ylabel('y')
            else:
                ax = plt.subplot(111, projection='3d')
                # 设置x轴、y轴和z轴使用科学计数法
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
                ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=1)
                for idx, (name, alg) in enumerate(self.algorithms.items()):
                    pop = alg.pop if pause or n_iter is None else alg.pop_history[n_iter]
                    objs = alg.objs if pause or n_iter is None else alg.objs_history[n_iter]
                    ax.scatter(pop[:, 0], pop[:, 1], objs.flatten(),
                               marker="o", s=50, c=self.colors[idx], zorder=2, label=name, alpha=0.6)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('obj')
        else:
            raise ValueError("The decision vector dimension must be less than 3 dimensions")
        plt.grid(True)
        plt.legend()
        if n_iter is None:
            plt.title("Objs-Decs")
        else:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()

    def plot_scores(self, n_iter=None, pause=False, pause_time=0.06):
        """绘制种群目标值"""
        plt.clf()
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        for idx, (name, alg) in enumerate(self.algorithms.items()):
            plt.plot(np.arange(len(alg.scores)), alg.scores,
                     marker=".", c=self.colors[idx], label=name, alpha=0.6)
        plt.xlabel('n_iter')
        plt.ylabel(self.score_type)
        plt.grid(True)
        plt.legend()
        if n_iter is None:
            plt.title(self.score_type + " Scores")
        else:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()
