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
import numpy as np


def save_array(arr, file_path, save_type='csv'):
    """将一个array进行保存"""
    if save_type == 'npy':
        np.save(file_path + '.' + save_type, arr)
    elif save_type == 'txt':
        np.savetxt(file_path + '.' + save_type, arr)
    elif save_type == 'csv':
        np.savetxt(file_path + '.' + save_type, arr, delimiter=',')
    else:
        raise ValueError(f'Cannot save as {save_type} type')
