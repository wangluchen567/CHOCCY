"""
保存工具类
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
    """将一个数组进行保存"""
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
    """将多个数组进行保存"""
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
    """保存为json文件"""
    with open(file_path + ".json", "w") as f:
        json.dump(data, f)


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


def get_timestamp(k=3):
    """获取当前时间点时间戳"""
    stamp = datetime.datetime.now().strftime("%m%d%H%M%S")
    # 为保证不冲突，加入一个k位随机数
    rn = np.random.randint(10 ** (k - 1), 10 ** k)
    timestamp = f"{stamp}{rn}"
    return timestamp
