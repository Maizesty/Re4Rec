def sum_third_column(file_path):
    total_sum = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            # 假设每列数据由空格分隔
            columns = line.split()
            if len(columns) >= 3:
                try:
                    # 将第三列转换为浮点数并累加
                    total_sum += float(columns[2])
                except ValueError:
                    # 如果转换失败，可以忽略这一行或者处理错误
                    print(f"无法转换的数据: {columns[2]}")
    return total_sum

# 使用示例
file_path = '/home/yssun/pytorch-fm/torchfm/model/test_fx/paper/data/motivation_ch4_rewrite_extract.txt'  # 替换为你的文件路径
result = sum_third_column(file_path)
print(f"第三列的总和是: {result}")
