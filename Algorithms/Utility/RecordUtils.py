"""
记录日志信息工具
Record Utils

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
import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(log_path=None, to_file=True, to_console=True, singleton=False):
    """
    配置日志记录器
    :param log_path: 输出日志的文件路径
    :param to_file: 是否输出日志到指定路径
    :param to_console: 是否输出日志到控制台
    :param singleton: 是否启用单例模式(默认不启用)
    """
    app_logger = AppLogger(
        log_path=log_path,
        to_file=to_file,
        to_console=to_console,
        singleton=singleton  # 单例模式开关
    )
    return app_logger.get_logger()


class AppLogger:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        # 获取单例模式开关
        singleton = kwargs.get('singleton', True)
        # 单例模式且实例已存在：直接返回现有实例
        if singleton and cls._instance is not None:
            return cls._instance
        # 创建新实例
        instance = super().__new__(cls)
        # 如果是单例模式：保存实例引用
        if singleton:
            cls._instance = instance
        # 返回实例
        return instance

    def __init__(self, log_path=None, to_file=True, to_console=True, singleton=True):
        """
        支持单例模式的日志类
        :param log_path: 输出日志的文件路径
        :param to_file: 是否输出日志到指定路径
        :param to_console: 是否输出日志到控制台
        :param singleton: 是否启用单例模式(默认启用)
        """
        self.singleton_mode = singleton  # 是否启用单例模式
        # 避免单例模式下的重复初始化
        if self.singleton_mode and self._initialized:
            return
        self.logger = None  # 日志记录对象
        # 使用参数或默认值
        self.log_path = log_path
        self.to_file = to_file
        self.to_console = to_console
        # 配置日志记录器
        self._setup_logger()
        # 单例模式下到此已初始化
        if self.singleton_mode:
            self._initialized = True

    def _setup_logger(self):
        """配置日志记录器"""
        # 根据单例模式选择logger名称
        if self.singleton_mode:
            # 创建 logger 对象
            # 单例模式：使用固定名称，确保全局唯一
            self.logger = logging.getLogger('app_singleton_logger')
        else:
            # 创建 logger 对象
            # 非单例模式：使用唯一名称，确保彼此独立
            self.logger = logging.getLogger(f'app_logger_{id(self)}')
            # 非单例模式：每次都重新配置 清除已有的 handler
            if self.logger.hasHandlers():
                self.logger.handlers.clear()
        # 设置日志级别为INFO
        # 只接收INFO以上级别，不包括DEBUG
        self.logger.setLevel(logging.INFO)
        # 创建一个格式器，日志格式为 [时间] 日志内容
        formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # 判断是否输出日志到指定文件
        if self.to_file and self.log_path is not None:
            # 检查日志文件路径，若不存在则自动创建日志目录
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # 创建 RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_path,
                maxBytes=3 * 1024 * 1024,  # 限制日志文件大小为 3MB
                backupCount=5,  # 保留 5 个备份文件
                encoding='utf-8'
            )
            # 设置文件处理器的日志级别
            file_handler.setLevel(logging.INFO)
            # 将格式器添加到文件处理器
            file_handler.setFormatter(formatter)
            # 将文件处理器添加到 logger
            self.logger.addHandler(file_handler)
        # 判断是否输出日志到控制台
        if self.to_console:
            # 创建一个控制台处理器，将日志输出到控制台
            console_handler = logging.StreamHandler()
            # 设置控制台处理器的日志级别
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """获取创建的logger对象"""
        return self.logger
