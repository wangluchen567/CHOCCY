import warnings
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class Artist:
    # 支持的绘图类型及其对应的ax方法
    _METHODS = {
        'line': 'plot',
        'scatter': 'scatter',
        'surface': 'plot_surface',
        'trisurf': 'plot_trisurf',
        'bar': 'bar',
        'contour': 'contour',
        'contourf': 'contourf',
        'fill_between': 'fill_between',
        # NetworkX 相关
        'nx_graph': 'nx_graph',  # 完整图
        'nx_nodes': 'nx_nodes',  # 节点
        'nx_edges': 'nx_edges',  # 边
        'nx_labels': 'nx_labels',  # 节点标签
        'nx_edge_labels': 'nx_edge_labels',  # 边标签
    }

    def __init__(self, draw_type, *args, **kwargs):
        """
        绘图元素 - 封装单个绘图指令

        :param draw_type: 绘图类型
        :param args: 接收‌不定数量的位置参数
        :param kwargs: 接收‌不定数量的关键字参数‌
        """
        self.draw_type = draw_type
        self.args = args
        self.kwargs = kwargs

    def draw_on(self, ax):
        """在指定坐标轴上绘制"""
        method_name = self._METHODS.get(self.draw_type)
        if not method_name:
            raise ValueError(f"Unsupported draw type: {self.draw_type}")

        method = getattr(ax, method_name)
        return method(*self.args, **self.kwargs)


class Frame:
    # 类级别的字体配置
    _font_configured = False
    _chinese_font_enabled = False

    @classmethod
    def enable_chinese_font(cls, enabled=True, font_family=None):
        """
        启用/禁用中文字体支持
        :param enabled: 是否启用中文字体
        :param font_family: 指定字体名称（可选，默认自动检测）
        """
        cls._chinese_font_enabled = enabled
        if enabled:
            cls._setup_font(font_family)
        else:
            cls._reset_font()

    @classmethod
    def _setup_font(cls, font_family=None):
        """配置中文字体（只执行一次）"""
        if cls._font_configured:
            return

        if font_family:
            plt.rcParams['font.sans-serif'] = [font_family]
        else:
            # 跨平台字体列表
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei', 'SimHei',  # Windows
                'PingFang SC', 'Hiragino Sans GB',  # macOS
                'WenQuanYi Zen Hei', 'Noto Sans CJK SC',  # Linux
                'DejaVu Sans'  # 回退
            ]
        plt.rcParams['axes.unicode_minus'] = False
        cls._font_configured = True

    @classmethod
    def _reset_font(cls):
        """重置字体为默认配置"""
        plt.rcParams['font.sans-serif'] = plt.rcParamsDefault['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = plt.rcParamsDefault['axes.unicode_minus']
        cls._font_configured = False

    def __init__(self, title="", is_3d=False, use_chinese_font=False, font_family=None):
        """
        绘图帧 - 包含一帧的所有绘图元素和设置

        :param title: 绘图标题
        :param is_3d: 是否绘制3d图像
        :param use_chinese_font: 是否使用中文字体
        :param font_family: 使用指定字体的名称
        """
        self.title = title
        self.is_3d = is_3d
        self.artists = []  # 绘图元素列表
        self._custom_artists = []  # 统一存储外部绘图函数
        self._applied_methods = []  # 存储方法配置 [(method_name, value), ...]

        # 设置字体
        if use_chinese_font:
            self.enable_chinese_font(True, font_family)

        # 延迟绘制的配置信息
        self._colorbar_configs = []  # 存储颜色条配置
        self._contour_label_configs = []  # 存储等高线标签配置

    # ---------- 合并帧对象 ----------

    def merge(self, other_frame, deep_copy=True):
        """
        将另一个帧的元素合并到当前帧（只合并元素，不合并图形设置）
        :param other_frame: 要合并的Frame对象
        :param deep_copy: 是否深拷贝（默认True，避免共享引用）
        :return: 合并后的当前Frame对象
        """
        # 合并 Artists
        for artist in other_frame.artists:
            if deep_copy:
                # 深拷贝 Artist
                new_artist = Artist(artist.draw_type, *artist.args, **artist.kwargs)
                self.artists.append(new_artist)
            else:
                self.artists.append(artist)
        # 合并自定义绘图（统一处理）
        # noinspection PyProtectedMember
        for custom_item in other_frame._custom_artists:
            if deep_copy:
                # noinspection PyProtectedMember
                self._custom_artists.append({
                    'func': custom_item['func'],
                    'args': custom_item['args'],
                    'kwargs': custom_item['kwargs'].copy()
                })
            else:
                # noinspection PyProtectedMember
                self._custom_artists.append(custom_item)
        return self

    @classmethod
    def merge_frames(cls, frames, deep_copy=True, use_first_settings=True):
        """
        合并多个帧（类方法）
        :param frames: Frame对象列表
        :param deep_copy: 是否深拷贝
        :param use_first_settings: 是否使用第一个帧的图形设置
        :return: 合并后的Frame对象
        """
        if not frames:
            return cls()
        # 以第一个帧为基础
        merged = cls(
            title=frames[0].title,
            is_3d=frames[0].is_3d,
            use_chinese_font=Frame._chinese_font_enabled
        )
        # 复制第一个帧的设置（使用统一的 _applied_methods）
        if use_first_settings:
            # noinspection PyProtectedMember
            for item in frames[0]._applied_methods:
                if deep_copy:
                    merged._applied_methods.append({
                        'name': item['name'],
                        'args': item.get('args', ()),
                        'kwargs': item.get('kwargs', {})
                    })
                else:
                    merged._applied_methods.append(item)
        # 合并所有帧的 Artists
        for frame in frames:
            merged.merge(frame, deep_copy=deep_copy)

        return merged

    # ---------- 添加绘图元素 ----------
    def add(self, draw_type, *args, **kwargs):
        """添加绘图元素，返回添加的Artist对象"""
        artist = Artist(draw_type, *args, **kwargs)
        self.artists.append(artist)
        return artist

    def add_custom(self, func, *args, **kwargs):
        """
        添加自定义绘图函数
        （支持任意基于 matplotlib 的第三方库，包括但不限于Matplotlib 原生、Seaborn、NetworkX等）

        :param func: 绘图函数，接受 ax 作为第一个或关键字参数
        :param args: 传递给绘图函数的参数
        :param kwargs: 传递给绘图函数的参数
        :return: self (支持链式调用)
        """
        self._custom_artists.append({
            'func': func,
            'args': args,
            'kwargs': kwargs
        })
        return self

    def add_line(self, x, y, z=None, **style):
        """添加线条 (2D/3D)，返回Artist对象"""
        args = (x, y, z) if z is not None else (x, y)
        return self.add('line', *args, **style)

    def add_scatter(self, x, y, z=None, **style):
        """添加散点 (2D/3D)，返回Artist对象"""
        args = (x, y, z) if z is not None else (x, y)
        return self.add('scatter', *args, **style)

    def add_surface(self, xs, ys, zs, **style):
        """添加3D曲面，返回Artist对象"""
        return self.add('surface', xs, ys, zs, **style)

    def add_trisurf(self, x, y, z, **style):
        """添加三角曲面 (3D)，返回Artist对象"""
        return self.add('trisurf', x, y, z, **style)

    def add_bar(self, x, height, **style):
        """添加柱状图，返回Artist对象"""
        return self.add('bar', x, height, **style)

    def add_contour(self, xs, ys, zs, **style):
        """添加等高线，返回Artist对象"""
        return self.add('contour', xs, ys, zs, **style)

    def add_contourf(self, xs, ys, zs, **style):
        """添加填充等高线，返回Artist对象"""
        return self.add('contourf', xs, ys, zs, **style)

    def add_fill_between(self, x, y1, y2=0, **kwargs):
        """添加填充区域，返回Artist对象"""
        return self.add('fill_between', x, y1, y2, **kwargs)

    def add_contour_labels(self, contour_artist, levels=None, fmt=None, **kwargs):
        """为等高线添加标签（延迟处理）"""
        self._contour_label_configs.append({
            'artist': contour_artist,
            'levels': levels,
            'fmt': fmt,
            'kwargs': kwargs
        })

    def add_colorbar(self, mappable_artist, **kwargs):
        """添加颜色条（延迟处理）"""
        self._colorbar_configs.append({
            'artist': mappable_artist,
            'kwargs': kwargs
        })

    # ---------- 图形设置 ----------
    def apply(self, *args, **kwargs):
        """
        注册一个或多个 Axes 方法调用

        这是一个通用接口，支持所有 matplotlib.axes.Axes 的公有方法。（包括 set_*、grid、legend 等），
        方法调用会按照注册顺序依次执行。

        示例:
            # 方式1：关键字参数（推荐，简洁直观）
                frame.apply(
                    set_xlim=(0, 10),           # 调用 ax.set_xlim(0, 10)
                    set_ylim=(-1, 1),           # 调用 ax.set_ylim(-1, 1)
                    set_title="My Plot",        # 调用 ax.set_title("My Plot")
                    grid=True,                  # 调用 ax.grid(True)
                    legend={'loc': 'upper right'},  # 调用 ax.legend(loc='upper right')
                    tick_params={'axis': 'both', 'direction': 'in'}  # 调用 ax.tick_params(**dict)
                )

            # 方式2：元组/列表方式（支持位置参数）
                frame.apply(
                    ('set_xlim', 0, 10),
                    ('set_ylim', -1, 1),
                    ('grid', True)
                )

            # 方式3：混合使用
                frame.apply(
                    ('set_xlim', 0, 10),
                    set_ylim=(-1, 1),
                    grid=True
                )
        :param args: 位置参数，支持以下格式：
            - 元组: ('method_name', arg1, arg2, ...)
            - 列表: ['method_name', arg1, arg2, ...]
        :param kwargs: 关键字参数，方法名作为键，参数作为值。
                       值可以是单个值、元组或列表，会自动解包为位置参数。

        :return: self（支持链式调用）
        """
        # 处理键值对
        for method_name, value in kwargs.items():
            if isinstance(value, dict):
                # 字典：作为关键字参数传递
                self._applied_methods.append({
                    'name': method_name,
                    'args': (),
                    'kwargs': value
                })
            elif isinstance(value, (tuple, list)):
                # 元组/列表：作为位置参数展开
                self._applied_methods.append({
                    'name': method_name,
                    'args': tuple(value),
                    'kwargs': {}
                })
            else:
                # 其他：作为单个位置参数
                self._applied_methods.append({
                    'name': method_name,
                    'args': (value,),
                    'kwargs': {}
                })
        # 处理位置参数
        for arg in args:
            if isinstance(arg, (tuple, list)) and len(arg) >= 1:
                method_name = arg[0]
                args_part = arg[1:] if len(arg) > 1 else ()
                self._applied_methods.append({
                    'name': method_name,
                    'args': args_part,
                    'kwargs': {}
                })
        return self

    def set_view(self, elev=30, azim=45):
        """设置3D视角"""
        return self.apply(view_init=(elev, azim))

    def set_limits(self, xlim=None, ylim=None, zlim=None):
        """设置坐标轴范围"""
        if xlim:
            self.apply(set_xlim=xlim)
        if ylim:
            self.apply(set_ylim=ylim)
        if zlim:
            self.apply(set_zlim=zlim)
        return self

    def set_labels(self, xlabel=None, ylabel=None, zlabel=None):
        """设置坐标轴标签"""
        if xlabel:
            self.apply(set_xlabel=xlabel)
        if ylabel:
            self.apply(set_ylabel=ylabel)
        if zlabel:
            self.apply(set_zlabel=zlabel)
        return self

    def set_grid(self, enable=True, **kwargs):
        """设置网格"""
        if kwargs:
            # 转换参数名：添加 grid_ 前缀
            tick_kwargs = {}
            for key, value in kwargs.items():
                tick_kwargs[f'grid_{key}'] = value
            # 通过 apply 直接调用 tick_params 设置样式
            self.apply(grid=enable, tick_params=tick_kwargs)
        else:
            self.apply(grid=enable)

        return self

    def set_title(self, title):
        """设置标题"""
        self.title = title
        return self.apply(set_title=title)

    def set_legend(self, show=True, **kwargs):
        """设置图例"""
        if show:
            return self.apply(legend=kwargs if kwargs else True)
        return self

    def set_yscale(self, scale_type='log', **kwargs):
        """设置y轴刻度类型"""
        if kwargs:
            return self.apply(set_yscale=(scale_type, kwargs))
        return self.apply(set_yscale=scale_type)

    def set_ticklabel_format(self, axis='both', style='sci', scilimits=(0, 0), **kwargs):
        """设置刻度标签格式"""
        return self.apply(ticklabel_format={
            'axis': axis,
            'style': style,
            'scilimits': scilimits,
            **kwargs
        })

    # noinspection PyPep8Naming
    def add_nx_graph(self, G, pos=None, **kwargs):
        """
        添加完整的 NetworkX 图（一次性绘制所有元素）
        :param G: NetworkX 图对象
        :param pos: pos: 节点位置字典（可选，默认使用 spring_layout）
        :param kwargs: 传递给 nx.draw_networkx 的参数
        """
        return self.add_custom(nx.draw_networkx, G, pos, **kwargs)

    # noinspection PyPep8Naming
    def add_nx_nodes(self, G, pos, **kwargs):
        """
        添加 NetworkX 节点
        :param G: NetworkX 图对象
        :param pos: 节点位置字典
        :param kwargs: 传递给 nx.draw_networkx_nodes 的参数
        """
        return self.add_custom(nx.draw_networkx_nodes, G, pos, **kwargs)

    # noinspection PyPep8Naming
    def add_nx_edges(self, G, pos, **kwargs):
        """
        添加 NetworkX 边
        :param G: NetworkX 图对象
        :param pos: 节点位置字典
        :param kwargs: 传递给 nx.draw_networkx_edges 的参数
        :return:
        """
        return self.add_custom(nx.draw_networkx_edges, G, pos, **kwargs)

    # noinspection PyPep8Naming
    def add_nx_labels(self, G, pos, labels=None, **kwargs):
        """
        添加节点标签
        :param G: NetworkX 图对象
        :param pos: 节点位置字典
        :param labels: 标签字典（可选，默认使用节点名）
        :param kwargs: 传递给 nx.draw_networkx_labels 的参数
        """
        kwargs = {**kwargs, **({'labels': labels} if labels is not None else {})}
        return self.add_custom(nx.draw_networkx_labels, G, pos, **kwargs)

    # noinspection PyPep8Naming
    def add_nx_edge_labels(self, G, pos, edge_labels=None, **kwargs):
        """
        添加边标签
        :param G: NetworkX 图对象
        :param pos: 节点位置字典
        :param edge_labels: 边标签字典（可选，默认使用边的字符串表示）
        :param kwargs: 传递给 nx.draw_networkx_edge_labels 的参数
        """
        kwargs = {**kwargs, **({'edge_labels': edge_labels} if edge_labels is not None else {})}
        return self.add_custom(nx.draw_networkx_edge_labels, G, pos, **kwargs)

    def add_sns_scatter(self, data, x=None, y=None, **kwargs):
        """添加 Seaborn 散点图"""
        return self.add_custom(sns.scatterplot, data=data, x=x, y=y, **kwargs)

    def add_sns_line(self, data, x=None, y=None, **kwargs):
        """添加 Seaborn 折线图"""
        return self.add_custom(sns.lineplot, data=data, x=x, y=y, **kwargs)

    def add_sns_bar(self, data, x=None, y=None, **kwargs):
        """添加 Seaborn 柱状图"""
        return self.add_custom(sns.barplot, data=data, x=x, y=y, **kwargs)

    def add_sns_hist(self, data, x=None, **kwargs):
        """添加 Seaborn 直方图"""
        return self.add_custom(sns.histplot, data=data, x=x, **kwargs)

    def add_sns_kde(self, data, x=None, **kwargs):
        """添加 Seaborn 核密度估计图"""
        return self.add_custom(sns.kdeplot, data=data, x=x, **kwargs)

    def add_sns_box(self, data, x=None, y=None, **kwargs):
        """添加 Seaborn 箱线图"""
        return self.add_custom(sns.boxplot, data=data, x=x, y=y, **kwargs)

    def add_sns_violin(self, data, x=None, y=None, **kwargs):
        """添加 Seaborn 小提琴图"""
        return self.add_custom(sns.violinplot, data=data, x=x, y=y, **kwargs)

    def add_sns_heatmap(self, data, **kwargs):
        """添加 Seaborn 热力图"""
        return self.add_custom(sns.heatmap, data=data, **kwargs)

    # ---------- 内部辅助方法 ----------
    def _apply_settings(self, ax):
        """应用所有已保存的图形设置"""
        for item in self._applied_methods:
            method_name = item['name']
            args = item.get('args', ())
            kwargs = item.get('kwargs', {})

            if hasattr(ax, method_name):
                method = getattr(ax, method_name)
                if callable(method):
                    method(*args, **kwargs)
            else:
                warnings.warn(f"Unknown method: {method_name}")

    def _apply_contour_labels(self, drawn_objects):
        """应用等高线标签"""
        for config in self._contour_label_configs:
            artist = config['artist']
            if id(artist) in drawn_objects:
                contour_obj = drawn_objects[id(artist)]
                levels = config['levels']
                fmt = config['fmt']
                kwargs = config['kwargs']
                if levels is not None:
                    contour_obj.clabel(levels, fmt=fmt, **kwargs)
                else:
                    contour_obj.clabel(fmt=fmt, **kwargs)

    def _apply_colorbars(self, drawn_objects, fig):
        """应用颜色条"""
        for config in self._colorbar_configs:
            artist = config['artist']
            if id(artist) in drawn_objects and fig:
                mappable = drawn_objects[id(artist)]
                fig.colorbar(mappable, **config['kwargs'])

    def _draw_custom_artists(self, ax):
        """
        绘制 自定义 图形对象
        :param ax: matplotlib 坐标轴对象
        """
        # 执行所有自定义绘图函数
        for custom in self._custom_artists:
            func = custom['func']
            args = custom['args']
            kwargs = custom['kwargs'].copy()
            # 智能注入 ax 参数
            if 'ax' not in kwargs:
                kwargs['ax'] = ax
            func(*args, **kwargs)

    # ---------- 绘制方法 ----------
    def draw_on(self, ax, fig=None, clear=True):
        """
        在坐标轴上绘制这一帧

        :param ax: matplotlib坐标轴对象
        :param fig: matplotlib图形对象（用于colorbar）
        :param clear: 是否清除坐标轴
        :return: 每个Artist对应的实际matplotlib对象
        """
        if clear:
            ax.clear()
        # 绘制所有图形
        drawn_objects = {}
        for artist in self.artists:
            result = artist.draw_on(ax)
            if result is not None:
                # 记录每个Artist对应的实际matplotlib对象
                drawn_objects[id(artist)] = result
        # 统一绘制所有自定义图形对象
        self._draw_custom_artists(ax)
        # 添加等高线标签
        self._apply_contour_labels(drawn_objects)
        # 添加颜色条
        self._apply_colorbars(drawn_objects, fig)
        # 应用图形设置
        self._apply_settings(ax)
        return drawn_objects

    def render_static(self, figsize=(6.4, 4.8), show=True):
        """将帧渲染为静态图像"""
        fig = plt.figure(figsize=figsize)
        if self.is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        self.draw_on(ax, fig=fig, clear=True)

        if show:
            plt.show()
        return fig, ax

    def save(self, filename, dpi=100):
        """保存帧为图片"""
        fig, _ = self.render_static(show=False)
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)
        print(f"Saved to: {filename}")


class Animator:

    def __init__(self, figsize=(6.4, 4.8), interval=30, save_frames=True, use_chinese_font=False, font_family=None):
        """
        动画器 - 支持实时显示和动画播放

        :param figsize: 绘图图像大小，元组 (width, height)
        :param interval: 动画帧间隔（毫秒）
        :param save_frames: 是否记录与存储帧
        :param use_chinese_font: 是否启用中文字体
        :param font_family: 指定中文字体名称（可选）
        """
        self.frames = []  # 存储所有帧
        self.fig = None  # 图形对象
        self.ax = None  # 坐标轴对象
        self.figsize = figsize  # 图形大小
        self.interval = interval  # 帧间隔（毫秒）
        self.save_frames = save_frames  # 是否记录与存储帧
        self.delay = self.interval / 1000.0  # 绘图延迟时间与interval同步
        self.use_chinese_font = use_chinese_font  # 是否启用中文字体
        self.font_family = font_family  # 中文字体名称
        self._realtime_mode = False  # 实时模式标志
        # 如果启用中文字体，在创建图形时设置
        if self.use_chinese_font:
            Frame.enable_chinese_font(True, self.font_family)

    # ---------- 配置方法 ----------
    def set_figsize(self, figsize):
        """
        设置绘图图像大小
        :param figsize: 元组 (width, height)
        """
        self.figsize = figsize
        if self.fig is not None:
            self.fig.set_size_inches(figsize)
        return self

    def set_interval(self, interval):
        """
        设置帧间隔（毫秒）
        :param interval: interval: 帧间隔（毫秒）
        """
        self.interval = interval
        # 绘图延迟时间与interval同步
        self.delay = self.interval / 1000.0
        return self

    def set_chinese_font(self, enabled=True, font_family=None):
        """
        设置中文字体支持
        :param enabled: 是否启用中文字体
        :param font_family: 指定字体名称（可选）
        :return:
        """
        self.use_chinese_font = enabled
        self.font_family = font_family if enabled else None
        if enabled:
            Frame.enable_chinese_font(True, font_family)
        else:
            Frame.enable_chinese_font(False)
        return self

    def set_save_frames(self, save_frames=True):
        """是否记录与存储帧"""
        self.save_frames = save_frames
        return self

    # ---------- 实时显示 ----------
    def start_realtime(self):
        """启动实时显示模式"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            plt.ion()  # 开启交互模式
            plt.show(block=False)
            plt.pause(0.001)  # 确保窗口显示
        self._realtime_mode = True
        return self

    def show(self, frame, animate=True):
        """
        显示一帧
        :param frame: Frame对象
        :param animate: 是否进行动画延迟
        """
        if not self._realtime_mode:
            self.start_realtime()
        # 根据save_frames决定是否保存帧
        if self.save_frames:
            self.frames.append(frame)
        # 判断是否需要重建坐标轴（2D/3D切换）
        need_recreate = False
        if self.ax is not None:
            is_current_3d = hasattr(self.ax, 'get_proj')
            is_new_3d = frame.is_3d
            if is_current_3d != is_new_3d:
                need_recreate = True
        # 重建坐标轴（如果需要）
        if need_recreate or self.ax is None:
            self.fig.clear()
            if frame.is_3d:
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
        # 绘制帧
        frame.draw_on(self.ax, clear=True)
        # 强制更新
        self._force_update()
        # 检查是否进行动画延迟
        if animate:
            # 控制动画延迟
            plt.pause(self.delay)
        else:
            # 若停止动画则关闭实时模式
            self.stop_realtime()
            plt.show()

    def _force_update(self):
        """强制更新图形"""
        if self.fig is not None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def stop_realtime(self):
        """停止实时模式"""
        self._realtime_mode = False
        if self.fig is not None:
            self.fig = None
            plt.ioff()

    def extract_frames(self, indices=None):
        """
        提取指定帧

        :param indices: 帧索引或索引列表
                     - None: 返回所有帧
                     - int: 返回单个帧
                     - list/slice: 返回帧列表
                     - tuple: 返回对应索引的帧列表
                     - 支持负数索引（从后往前）
        :return: 提取指定的帧
        """
        if not self.frames:
            return None if isinstance(indices, int) else []

        # 处理不同的索引类型
        if indices is None:
            # 返回所有帧的副本
            return self.frames.copy()

        elif isinstance(indices, int):
            # 返回单个帧
            return self.frames[indices]

        elif isinstance(indices, slice):
            # 返回切片范围
            return self.frames[indices]

        elif isinstance(indices, (list, tuple)):
            # 返回列表/元组中指定索引的帧
            return [self.frames[i] for i in indices]

        else:
            raise TypeError(f"Unsupported index type: {type(indices)}")

    # ---------- 动画播放 ----------
    def replay(self, frames=None, interval=None, repeat=False, save_path=None, dpi=100, save_fps=None, fps_factor=0.1):
        """
        播放已保存的动画

        :param frames: 帧列表（可以使用extract_frames提取），默认全部帧
        :param interval: interval: 帧间隔(ms)，None则使用默认值
        :param repeat: 是否重复播放
        :param save_path: 保存路径，None则不保存
        :param dpi: 保存时的清晰度（默认100，建议150-300）
        :param save_fps: 保存时的帧率，None则自动估算
        :param fps_factor: 实际帧率与理论帧率的比例因子（默认0.1）
                           理论帧率 = 1000 / interval
                           实际帧率 ≈ 理论帧率 * fps_factor
        :return: 动画实例
        """
        if frames is None:
            frames = self.frames
        if not frames or len(frames) == 0:
            warnings.warn("No frames to replay")
            return None
        # 使用传入的interval或默认的self.interval
        play_interval = interval if interval is not None else self.interval
        # 估算保存帧率
        if save_fps is None:
            theoretical_fps = 1000 / play_interval
            save_fps = max(1, int(theoretical_fps * fps_factor))
        # 创建播放用的图形
        play_fig = plt.figure(figsize=self.figsize, dpi=dpi)

        # 创建更新函数
        def update(frame_idx):
            """更新函数"""
            play_fig.clear()
            frame = frames[frame_idx]
            # 创建对应类型的坐标轴
            if frame.is_3d:
                ax = play_fig.add_subplot(111, projection='3d')
            else:
                ax = play_fig.add_subplot(111)
            # 绘制帧
            frame.draw_on(ax, clear=False)
            return []

        # 创建动画
        ani = FuncAnimation(
            play_fig, update,
            frames=len(frames),
            interval=play_interval,
            blit=False,
            repeat=repeat
        )
        # 保存动画
        if save_path is not None:
            self._save_animation(ani, save_path, save_fps, dpi)
        # 展示结果
        plt.show()
        return ani

    @staticmethod
    def _save_animation(animation, filename, fps=1000 / 30, dpi=100):
        """保存动画到文件"""
        try:
            if filename.endswith('.gif'):
                animation.save(filename, writer='pillow', fps=fps, dpi=dpi)
            elif filename.endswith('.mp4'):
                animation.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
            else:
                animation.save(filename, writer='pillow', fps=fps, dpi=dpi)
            print(f"Animation saved: {filename} (fps={fps:.1f}, dpi={dpi})")
        except Exception as e:
            warnings.warn(f"Failed to save animation: {e}")

    def clear(self):
        """清除所有帧"""
        self.frames.clear()
        return self
