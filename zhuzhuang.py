import numpy as np
import matplotlib.pyplot as plt

# 组别和子组别的标签
groups = ['FLARE22', 'TCIA', 'AMOS','Synapse']
subgroups = ['client_BC', 'w/o A', 'client_AC', 'w/o B', 'client_AB', 'w/o C', 'client_ABC']

# 每个子组别的数据
data = np.array([
    [75.53, 81.23, 81.86, 84.80, 85.75, 86.67, 86.61],
    [49.49, 60.59, 55.09, 58.32, 59.02, 63.29, 65.36],
    [51.83, 53.66, 49.99, 49.23, 40.20, 54.62, 52.97],
    [68.20, 73.32, 69.93, 77.64, 66.16, 74.57, 77.22]
])
# color = ['#DDF2C6', '#C0D1DC', '#CEEECF', '#CFCCE3', '#BBE3DA', '#BBA8BF', '#F19C99']

color = ['#DDF2C6', '#C0D1DC', '#C8F2E9', '#CFCCE3', '#AED4CB', '#BBA8BF', '#F19C99']
# color = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#98df8a', '#d62728', '#ff9896']
# 设置图纸的尺寸
plt.figure(figsize=(9, 5))
# plt.grid(True, axis='y', linestyle='--', alpha=0.5)
# 设置柱状图的宽度和间隔
bar_width = 0.08
group_spacing = 0.3
subgroup_spacing = 0.02

# 计算每个组的位置
index = np.arange(len(groups))

# 计算每个子组的位置
subgroup_indices = (index[:, np.newaxis] +
                    (bar_width + subgroup_spacing) * np.arange(len(subgroups)))

# 将subgroup1和subgroup2靠在一起
subgroup_indices[:, 0] += subgroup_spacing

subgroup_indices[:, 2] += subgroup_spacing

subgroup_indices[:, 4] += subgroup_spacing

subgroup_indices[:, 6] += subgroup_spacing

# 绘制柱状图
for i, subgroup in enumerate(subgroups):
    subgroup_data = data[:, i]
    plt.bar(subgroup_indices[:, i], subgroup_data, bar_width, label=subgroup,color = color[i])

# 设置x轴标签和标题
plt.xlabel('Dataset')
plt.ylabel('Dice Value (%)')
# plt.title('Nested Grouped Bar Chart')

# 设置x轴刻度标签
plt.xticks(index + group_spacing, groups)

# 添加图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=len(subgroups))

plt.savefig(r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\show\zhuzhuang.png')
# 展示图表
plt.show()





