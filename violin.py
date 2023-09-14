import matplotlib.pyplot as plt
import numpy as np


# 生成13个随机数据集
np.random.seed(0)
num_models = 13
# data = [np.random.normal(0, 1, 1000) for _ in range(num_models-1)]
# 读取.npy文件
dice_scores_1 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_1.npy')
dice_scores_2 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_2.npy')
dice_scores_3 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_3.npy')
dice_scores_4 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_4.npy')
dice_scores_5 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_5.npy')
dice_scores_6 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_6.npy')
dice_scores_7 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_7.npy')
dice_scores_8 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_8.npy')
dice_scores_9 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_9.npy')
dice_scores_10 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_10.npy')
dice_scores_11 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_11.npy')
dice_scores_12 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data_12.npy')
dice_scores_13 = np.load('D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dice_data\dice_data.npy')

data = [dice_scores_1,dice_scores_2,dice_scores_3,dice_scores_4,dice_scores_5,dice_scores_6,dice_scores_7,dice_scores_8,dice_scores_9,dice_scores_10,dice_scores_11,dice_scores_12,dice_scores_13]
# Define different colors for each model (e.g., using a color map)
colors = plt.cm.viridis(np.linspace(0, 1, num_models))# You can choose any colormap

# 创建子图
fig, ax = plt.subplots()

# 绘制多个小提琴图并设置不同的颜色
violins = ax.violinplot(data, showmeans=True, showmedians=False, points=100,widths=0.8,showextrema=False)
for i, violin in enumerate(violins['bodies']):
    violin.set_facecolor(colors[i])


# 添加箱线图（可选，根据需要显示）
for i, d in enumerate(data):
    positions = [i + 1]
    bp = ax.boxplot([d], positions=positions, showfliers=False, widths=0.85)

# 添加坐标轴标签和标题
ax.set_ylabel('Dice Value')

# 设置x轴标签
model_names = ['U-Net','UNet++','UNet 3+','M2SNet','CANet','LeViT-UNet','TransUNet','MISSFormer','TransAttUnet',
'Swin-Unet','UCTransNet', 'MedFormer','Ours']
ax.set_xticks(np.arange(1, num_models+1))
ax.set_xticklabels(model_names,rotation=45)
plt.subplots_adjust(left=0.1,bottom=0.2)
plt.savefig(r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\show\violin.png')
plt.show()
