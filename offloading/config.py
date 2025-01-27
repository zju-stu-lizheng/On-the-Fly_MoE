def get_gpu_layers(value):
    """
    根据输入的整数返回对应的列表。
    
    参数:
    value (int): 输入的整数，可能的值为 1, 2, 4, 6, 7。
    
    返回:
    list: 根据输入值返回的列表。
    """
    if value == 1:
        return [0]
    elif value == 2:
        return [0, 1]
    elif value == 4:
        return [0, 1, 2, 3]
    elif value == 6:
        return [0, 1, 2, 3, 6, 8]
    elif value == 7:
        return [0, 1, 2, 3, 6, 8, 31]
    else:
        raise ValueError("输入的值必须是 1, 2, 4, 6 或 7 中的一个。")