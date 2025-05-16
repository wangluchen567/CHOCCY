"""
读取文件工具
Read Utils

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
import json
import pickle
import numpy as np


def load_array(file_path):
    """加载保存的一个数组的文件"""
    # 根据文件扩展名判断文件类型
    if file_path.endswith('.npy'):
        # 加载 npy 文件
        arr = np.load(file_path)
    elif file_path.endswith('.txt'):
        # 加载 txt 文件
        arr = np.loadtxt(file_path)
    elif file_path.endswith('.csv'):
        # 加载 csv 文件
        arr = np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return arr


def load_arrays(file_path):
    """加载保存的多个数组的文件"""
    # 根据文件扩展名判断文件类型
    if file_path.endswith('.npz'):
        # 加载 npz 文件
        loaded_data = np.load(file_path)
        # 将数据转换为字典
        arrays_dict = {key: loaded_data[key] for key in loaded_data.files}
        loaded_data.close()  # 关闭文件
    elif file_path.endswith('.json'):
        # 加载 json 文件
        with open(file_path, "r") as f:
            loaded_json_data = json.load(f)
        # 将列表转换回 numpy 数组
        arrays_dict = {key: np.array(value) for key, value in loaded_json_data.items()}
    elif file_path.endswith('.pkl'):
        # 加载 pkl 文件
        with open(file_path, "rb") as f:
            arrays_dict = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return arrays_dict
