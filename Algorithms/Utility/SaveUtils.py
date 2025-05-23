"""
保存文件工具
Save Utils

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
import datetime
import numpy as np


def save_array(arr, file_path, save_type='csv'):
    """
    将一个数组进行保存
    :param arr: numpy数组
    :param file_path: 保存文件路径
    :param save_type: 保存文件类型
    """
    if save_type == 'npy':
        # 保存为 npy 文件
        np.save(file_path + '.' + save_type, arr)
    elif save_type == 'txt':
        # 保存为 txt 文件
        np.savetxt(file_path + '.' + save_type, arr)
    elif save_type == 'csv':
        # 保存为 csv 文件
        np.savetxt(file_path + '.' + save_type, arr, delimiter=',')
    else:
        raise ValueError(f"Unsupported save type: {save_type}")


def save_arrays(arrays_dict: dict, file_path, save_type='npz'):
    """
    将多个数组进行保存
    :param arrays_dict: numpy数组组成的字典
    :param file_path: 保存文件路径
    :param save_type: 保存文件类型
    """
    if save_type == 'npz':
        # 保存为 npz 文件
        np.savez(file_path, **arrays_dict)
    elif save_type == 'json':
        # 保存为 json 文件 (需要转换为list)
        json_data = {key: value.tolist() for key, value in arrays_dict.items()}
        with open(file_path + ".json", "w") as f:
            json.dump(json_data, f)
    elif save_type == 'pkl':
        # 保存为 pkl 文件
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(arrays_dict, f)
    else:
        raise ValueError(f"Unsupported save type: {save_type}")


def save_json(data: dict, file_path):
    """
    保存为json文件
    :param data: 要保存的数据
    :param file_path: 文件路径
    """
    with open(file_path + ".json", "w") as f:
        json.dump(data, f)


def get_timestamp():
    """获取当前时间点时间戳(月日时分秒毫秒)"""
    # 获取当前时间
    now_time = datetime.datetime.now()
    # 获取微秒值并转换为毫秒值（微秒值除以1000）
    milliseconds = now_time.microsecond // 1000
    # 得到时间戳(月日时分秒毫秒)
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S") + f"{milliseconds:03d}"
    return timestamp
