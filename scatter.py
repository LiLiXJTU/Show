import matplotlib.pyplot as plt

# 横坐标（路线）
models = ['LeViT-UNet', 'TransUNet', 'MISSFormer', 'TransAttUnet', 'Swin-Unet','UCTransNet','MedFormer','Ours']
parameter_sizes = [52.15*20, 105.28*20, 42.46*20, 25.97*20, 41.38*20,66.44*20, 28.07*20,24.88*20]
# parameter_sizes = [52.15, 105.28, 42.46, 25.97, 41.38,66.44, 28.07,24.88]

# MSD数据集上的Diceneiro
msd_dice_scores = [76.86, 81.14, 75.87, 80.70, 76.59,77.49,81.94,82.83]

# NCSLS数据集上的Dice
ncsls_dice_scores = [56.29, 60.15, 58.48, 59.04,58.72,57.87, 59.97,62.99]

# 绘制折线图
s1 = plt.scatter(models, msd_dice_scores, s=parameter_sizes, alpha=0.5,label='MSD',color = '#B39CD0')
s2 = plt.scatter(models, ncsls_dice_scores, s=parameter_sizes, alpha=0.5,label='NCSLS', color = '#FF8066')
# plt.plot(models, msd_dice_scores, marker='o', label='MSD')
# plt.plot(models, ncsls_dice_scores, marker='o', label='NCSLS')
plt.xticks(rotation=45)
# 设置图例
plt.ylim(53,86)
# 设置图形标题和坐标轴标签
plt.title('Dice Scores Comparison')
plt.xlabel('Transformer-based Models')
plt.ylabel('Dice Score (%)')
plt.tight_layout()
# 显示图形
for i in range(len(models)):
    plt.text(models[i], msd_dice_scores[i], str(msd_dice_scores[i]), ha='center', va='center')
    plt.text(models[i], ncsls_dice_scores[i], str(ncsls_dice_scores[i]), ha='center', va='center')
plt.savefig(r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\show\scatter.png')
plt.show()
