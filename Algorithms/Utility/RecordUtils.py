"""
记录信息工具类
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
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(log_path=None, to_file=False, to_console=True):
    """
    配置日志记录器
    :param log_path: 输出日志的文件路径
    :param to_file: 是否输出日志到指定路径
    :param to_console: 是否输出日志到控制台
    """
    # 创建 logger 对象
    logger = logging.getLogger()
    # 设置日志级别为INFO
    # 只接收INFO以上级别，不包括DEBUG
    logger.setLevel(logging.INFO)
    # 创建一个格式器，日志格式为 [时间] 日志内容
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 判断是否输出日志到指定文件
    if to_file:
        if log_path is None:
            raise ValueError("Please specify the log output path !")
        # 创建 RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=3 * 1024 * 1024,  # 限制日志文件大小为 3MB
            backupCount=5,  # 保留 5 个备份文件
            encoding='utf-8'
        )
        # 设置文件处理器的日志级别
        file_handler.setLevel(logging.INFO)
        # 将格式器添加到文件处理器
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到 logger
        logger.addHandler(file_handler)
    # 判断是否输出日志到控制台
    if not to_file and to_console:
        # 创建一个控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        # 设置控制台处理器的日志级别
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
