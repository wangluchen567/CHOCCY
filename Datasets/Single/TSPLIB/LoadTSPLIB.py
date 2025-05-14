"""
加载TSPLIB数据集的函数
Load TSPLIB Functions

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
import numpy as np


def load_euc_2d(file_path):
    """加载给定城市点坐标位置的数据集"""
    # 定义一个空字典来存储文件中的信息
    data = {
        'name': None,
        'type': None,
        'comment': None,
        'dimension': None,
        'edge_weight_type': None,
        'node_coord': [],
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


def load_matrix(file_path):
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
            parts_data = [float(parts[i]) for i in range(len(parts))]
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
    return data
