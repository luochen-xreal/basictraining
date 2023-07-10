import matplotlib.pyplot as plt
import numpy as np

def reprojectionErrorScatterPlot(x, y, indices):
    colors = np.arange(len(indices))  # 生成与索引长度相同的连续整数数组
    plt.scatter(x, y, marker='x', c=colors, cmap='rainbow')  # 设置标记类型为 "X"，使用 'viridis' 颜色映射
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reprojection Error')
    plt.colorbar(label='Index')  # 添加颜色映射的标签
    plt.show()


def readDataFromFile(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.strip().split()
            data.append((float(x), float(y)))
    return data

filename = '/home/xreal/xreal/project/build/data.txt'
data = readDataFromFile(filename)
# print(data)
x = [point[0] for point in data]
y = [point[1] for point in data]
indices = [i for i in range(len(data))]
# print(indices)

reprojectionErrorScatterPlot(x, y, indices)
