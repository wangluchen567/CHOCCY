# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
格式化工具模块（函数集合）
"""


def format_as_table(data: dict,
                    row_key: str,
                    col_key: str,
                    transpose: bool = False,
                    float_format: str = ".6e"):
    """
    通用的数据表格格式化函数

    :param data: 字典格式的数据，格式为 {列名: [值列表]}
    :param row_key: 转置前行标签所在的列名
    :param col_key: 转置后行标签所在的列名
    :param transpose: 是否转置；
                      False: 保持原始方向，行标签 = data[row_key]
                      True:  转置表格，行标签 = data[col_key]（此时col_key必须在data中存在）
    :param float_format: 浮点数的格式化格式，默认为 ".6e"（科学计数法，6位小数）
                      常用格式：
                      - ".6e" : 科学计数法，6位小数
                      - ".4e" : 科学计数法，4位小数
                      - ".2f" : 定点数，2位小数
                      - ".6f" : 定点数，6位小数
                      - ".3g" : 通用格式，3位有效数字
    :return: 格式化后的表格字符串
    """
    if not data:
        raise ValueError("No data available")

    # 根据是否转置确定索引列和左上角文字
    if transpose:
        data = transpose_data(data, row_key, col_key)
        index_col = col_key  # 转置后，col_key 成为行标签列
        top_left = col_key  # 转置后，col_key 显示在左上角
    else:
        index_col = row_key  # 不转置，row_key 是行标签列
        top_left = row_key  # 不转置，row_key 显示在左上角

    # 获取行标签和数据列
    row_labels = data[index_col]
    data_cols = [k for k in data.keys() if k != index_col]

    # 确定每列的宽度
    col_widths = {}

    # 第一列宽度
    first_col_width = len(top_left)
    for label in row_labels:
        first_col_width = max(first_col_width, len(str(label)))
    col_widths['first'] = first_col_width + 2

    # 其他列宽度
    for col in data_cols:
        max_width = len(col)
        for val in data[col]:
            formatted = format_value(val, float_format)
            max_width = max(max_width, len(formatted))
        col_widths[col] = max_width + 2

    # 构建表格
    lines = []

    # 表头行
    header_parts = [f"{top_left:<{col_widths['first']}}"]
    for col in data_cols:
        header_parts.append(f"{col:^{col_widths[col]}}")
    header_line = " ".join(header_parts)

    # 分隔线
    separator_line = "-" * len(header_line)

    # 头部
    lines.append("=" * len(header_line))
    lines.append(header_line)
    lines.append(separator_line)

    # 数据行
    for i, label in enumerate(row_labels):
        row_parts = [f"{str(label):<{col_widths['first']}}"]
        for col in data_cols:
            val = data[col][i]
            formatted = format_value(val, float_format)
            row_parts.append(f"{formatted:>{col_widths[col]}}")
        lines.append(" ".join(row_parts))

    # 尾部
    lines.append("=" * len(header_line))

    return "\n".join(lines)


def format_value(value, float_format: str = ".6e"):
    """
    格式化单个值（数字用科学计数法，字符串保持不变）

    :param value: 任意类型的值
    :param float_format: 浮点数的格式化格式，默认为 ".6e"（科学计数法，6位小数）
                          常用格式：
                          - ".6e" : 科学计数法，6位小数
                          - ".4e" : 科学计数法，4位小数
                          - ".2f" : 定点数，2位小数
                          - ".6f" : 定点数，6位小数
                          - ".3g" : 通用格式，3位有效数字
    :return: 格式化后的字符串
    """
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            value = float(value)
        # 使用传入的格式字符串
        return f"{value:{float_format}}"
    else:
        return str(value)


def transpose_data(data: dict, row_key: str, col_key: str):
    """
    转置数据结构，使用 col_key 作为新索引列
    将原始数据 {列名: [值列表]} 转换为 {新行名: [值列表]}

    :param data: 原始数据字典，格式为 {列名: [值列表]}
    :param row_key: 转置前行标签所在的列名
    :param col_key: 转置后行标签所在的列名
    :return: 转置后的新数据字典
    """
    # 获取行标识列表
    row_values = data[row_key]
    other_cols = [k for k in data.keys() if k != row_key]

    # 重新组织数据
    transposed = {col_key: other_cols}
    for i, val in enumerate(row_values):
        transposed[val] = [data[col][i] for col in other_cols]

    return transposed
