# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
数据保存器(函数)集合
"""

import os
import csv
import json
import pickle
import numpy as np
from .formatter import transpose_data


def save_to_file(data, file_path: str, file_format: str = None) -> None:
    """
    数据保存函数，保存数据到指定文件中

    :param data: 字典格式的数据
    :param file_path: 文件路径
    :param file_format: 支持的格式 'csv', 'json', 'pkl', 'npz'
    :return: None
    """
    # 自动判断格式
    if file_format is None:
        # 检查是否有扩展名
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.npz':
            file_format = 'npz'
        elif ext == '.json':
            file_format = 'json'
        elif ext == '.pkl':
            file_format = 'pkl'
        elif ext == '.csv':
            # .csv后缀时，去掉后缀作为文件夹
            file_path = os.path.splitext(file_path)[0]
            file_format = 'csv'
        else:
            # 无后缀或未知后缀，默认为csv（文件夹模式）
            file_format = 'csv'

    # 支持保存的格式
    supported_formats = {'csv', 'json', 'pkl', 'npz'}

    # 确保目录存在（文件模式）
    if file_format != 'csv':
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    if file_format == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    elif file_format == 'pkl':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    elif file_format == 'npz':
        # 转换为numpy可保存的格式
        npz_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # 字典类型：保持原样
                npz_data[key] = value
            elif isinstance(value, list):
                # 列表类型：转换为numpy数组
                npz_data[key] = np.array(value) if value is not None else None
            else:
                npz_data[key] = value
        np.savez_compressed(file_path, **npz_data)

    elif file_format == 'csv':
        # 创建文件夹
        os.makedirs(file_path, exist_ok=True)
        for key, value in data.items():
            if isinstance(value, dict):
                # 字典类型：保存为{key}.csv
                _save_dict_to_csv(value, os.path.join(file_path, f'{key}.csv'))
            elif isinstance(value, list) and value:
                # 列表类型：保存为数组
                arr = np.array(value)
                if arr.size > 0:
                    np.savetxt(os.path.join(file_path, f'{key}.csv'), arr, delimiter=',')
    else:
        raise ValueError(
            f"Unsupported file format: '{file_format}'. "
            f"Supported formats: {supported_formats}"
        )


def _save_dict_to_csv(data: dict, file_path: str) -> None:
    """
    将字典保存为CSV文件

    :param data: 字典数据
    :param file_path: CSV文件路径
    """
    if not data:
        return

    # 明确转换为列表确保顺序一致
    keys = list(data.keys())
    values = [data[k] for k in keys]

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(keys)
        # 写入值
        writer.writerow(values)


def save_as_table(data: dict,
                  csv_path: str,
                  row_key: str,
                  col_key: str,
                  transpose: bool = False,
                  float_format: str = ".6e"):
    """
    将数据保存为CSV表格文件

    :param data: 字典格式的数据，格式为 {列名: [值列表]}
    :param csv_path: CSV文件保存路径
    :param row_key: 转置前行标签所在的列名
    :param col_key: 转置后行标签所在的列名
    :param transpose: 是否转置；
                      False: 保持原始方向，行标签 = data[row_key]
                      True:  转置表格，行标签 = data[col_key]（此时col_key必须在data中存在）
    :param float_format: 浮点数的格式化格式，默认为 ".6e"
                         想要高精度可以设置 ".10e" 或 ".12f"
                         想要原始效果可以设置 ".15g"
    :return: 是否保存成功
    """
    if not data:
        print(f"Warning: No data to save to {csv_path}")
        return False

    # 确保目录存在
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 根据是否转置确定索引列和左上角文字
    if transpose:
        data = transpose_data(data, row_key, col_key)
        index_col = col_key  # 转置后，col_key 成为行标签列
        top_left = col_key  # 转置后，col_key 显示在左上角
    else:
        index_col = row_key  # 不转置，row_key 是行标签列
        top_left = row_key  # 不转置，row_key 显示在左上角

    # 准备CSV数据
    row_labels = data[index_col]
    data_cols = [k for k in data.keys() if k != index_col]

    # 构建CSV数据
    csv_data = [[top_left] + data_cols]

    # 数据行
    for i, label in enumerate(row_labels):
        row = [str(label)]
        for col in data_cols:
            val = data[col][i]
            if isinstance(val, (int, float)):
                if isinstance(val, int):
                    val = float(val)
                formatted = f"{val:{float_format}}"
            else:
                formatted = str(val)
            row.append(formatted)
        csv_data.append(row)

    # 保存CSV文件
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        print(f"Table saved to: {csv_path}")
        return True

    except Exception as e:
        print(f"Error saving table to {csv_path}: {e}")
        return False
