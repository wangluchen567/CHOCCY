# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
数据加载器(函数)集合
"""

import os
import json
import pickle
import numpy as np


def load_from_file(file_path: str, file_format: str = None) -> dict:
    """
    数据加载函数

    :param file_path: 文件路径或文件夹路径（CSV格式时）
    :param file_format: 支持的格式 'csv', 'json', 'pkl', 'npz'
    :return: 数据字典
    """
    # 支持读取的格式
    supported_formats = {'csv', 'json', 'pkl', 'npz'}

    # 自动检测格式
    if file_format is None:
        if os.path.isdir(file_path):
            file_format = 'csv'
        elif file_path.endswith('.npz'):
            file_format = 'npz'
        elif file_path.endswith('.json'):
            file_format = 'json'
        elif file_path.endswith('.pkl'):
            file_format = 'pkl'
        else:
            raise ValueError(f"Cannot detect format from: {file_path}")

    if file_format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif file_format == 'pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    elif file_format == 'npz':
        with np.load(file_path, allow_pickle=True) as npz:
            data = {}
            for key in npz.keys():
                value = npz[key]
                if key == 'metrics' or isinstance(value, np.ndarray) and value.dtype == np.dtype('O'):
                    # 可能是字典类型
                    if value.size == 1:
                        data[key] = value.item()
                    else:
                        data[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                else:
                    data[key] = value
        return data

    elif file_format == 'csv':
        # CSV格式：读取文件夹下所有csv文件
        data = {}
        for filename in os.listdir(file_path):
            if not filename.endswith('.csv'):
                continue

            key = filename[:-4]  # 去掉.csv后缀
            csv_path = os.path.join(file_path, filename)

            # 先尝试矩阵格式
            try:
                # 读取时确保读取的是2维数据（ndmin=2）
                data[key] = np.loadtxt(csv_path, delimiter=',', ndmin=2).tolist()
            except ValueError:
                # 矩阵失败，尝试字典格式（2行文件）
                with open(csv_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]

                if len(lines) == 2:
                    headers = lines[0].split(',')
                    values = lines[1].split(',')
                    data[key] = {h: float(v) for h, v in zip(headers, values)}
                else:
                    raise ValueError(
                        f"Cannot parse {csv_path}: not matrix (np.loadtxt failed) and not dict (need 2 lines)")

        return data

    else:
        raise ValueError(
            f"Unsupported file format: '{file_format}'. "
            f"Supported formats: {supported_formats}"
        )


def load_tsp_coord(file_path):
    """加载给定城市点坐标位置的数据集"""
    # 定义一个空字典来存储文件中的信息
    data = {
        'name': None,
        'type': None,
        'comment': None,
        'dimension': None,
        'edge_weight_type': None,
        'node_coord': [],
        'dist_matrix': None
    }
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 用于存储当前正在解析的节
    current_section = None
    # 逐行解析文件内容
    for line in lines:
        # 去除行尾的换行符
        line = line.strip()
        # 忽略空行
        if not line:
            continue
        if line == 'EOF':
            # 若到结尾则停止
            break
        # 检查是否是节的标题
        elif (line.startswith('NAME') or
              line.startswith('TYPE') or
              line.startswith('COMMENT') or
              line.startswith('DIMENSION') or
              line.startswith('EDGE_WEIGHT_TYPE')):
            key, value = line.split(':')
            key = key.strip().lower()  # 将键转换为小写
            if key in ['dimension']:
                data[key] = int(value)  # 转换为整数
            else:
                data[key] = value.strip()  # 去除两端空白字符
        elif line == 'NODE_COORD_SECTION':
            current_section = 'node_coord'
        elif current_section == 'node_coord':
            parts = line.split()
            if len(parts) >= 3:  # 确保行不为空且有足够的数据
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                data['node_coord'].append([x, y])
        else:
            continue
    # 将点坐标位置数据转换为numpy数据
    data['node_coord'] = np.array(data['node_coord'])
    return data


def load_tsp_matrix(file_path):
    """加载给定城市点之间的距离矩阵的数据集"""
    # 定义一个空字典来存储文件中的信息
    data = {
        'name': None,
        'type': None,
        'comment': None,
        'dimension': None,
        'edge_weight_type': None,
        'node_coord': [],
        'dist_matrix': [],
    }
    # 初始化一个下三角矩阵信息
    lower_triangle_data = []
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 用于存储当前正在解析的节
    current_section = None
    # 逐行解析文件内容
    for line in lines:
        # 去除行尾的换行符
        line = line.strip()
        # 忽略空行
        if not line:
            continue
        if line == 'EOF':
            # 若到结尾则停止
            break
        # 检查是否是节的标题
        elif (line.startswith('NAME') or
              line.startswith('TYPE') or
              line.startswith('COMMENT') or
              line.startswith('DIMENSION') or
              line.startswith('EDGE_WEIGHT_TYPE')):
            key, value = line.split(':')
            key = key.strip().lower()  # 将键转换为小写
            if key in ['dimension']:
                data[key] = int(value)  # 转换为整数
            else:
                data[key] = value.strip()  # 去除两端空白字符
        elif line == 'EDGE_WEIGHT_SECTION':
            current_section = 'dist_matrix'
        elif line == 'DISPLAY_DATA_SECTION':
            current_section = 'node_coord'
        elif current_section == 'dist_matrix':
            parts = line.split()
            parts_data = list(map(float, parts))
            lower_triangle_data.extend(parts_data)
        elif current_section == 'node_coord':
            parts = line.split()
            if len(parts) >= 3:  # 确保行不为空且有足够的数据
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                data['node_coord'].append([x, y])
        else:
            continue
    # 城市数量
    n = data['dimension']
    index = 0  # 数据下标
    data['dist_matrix'] = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):  # 遍历下三角（含对角线）
            data['dist_matrix'][i][j] = lower_triangle_data[index]
            data['dist_matrix'][j][i] = lower_triangle_data[index]
            index += 1
    # 将点坐标位置数据转换为numpy数据
    data['node_coord'] = np.array(data['node_coord'])
    if len(data['node_coord']) == 0:
        data['node_coord'] = None
    if len(data['dist_matrix']) == 0:
        data['dist_matrix'] = None
    return data
