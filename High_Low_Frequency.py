import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('D:\data\FDG-PET-CT-Lesions\CT_show/33.png', cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 将图像转换为浮点型并进行归一化
image_float = image.astype(np.float32)
normalized_image = cv2.normalize(image_float, None, 0, 1, cv2.NORM_MINMAX)

# 定义池化窗口大小
ksize = (3, 3)  # 池化窗口的大小

# 应用均值池化
smoothed_image = cv2.blur(normalized_image, ksize)

# 计算高频信息图
high_pass_image = normalized_image - smoothed_image

# 计算低频信息图
low_pass_image = smoothed_image

# 显示高频信息图和低频信息图
plt.subplot(1, 3, 1), plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(high_pass_image, cmap='gray')
plt.title('High Frequency'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(low_pass_image, cmap='gray')
plt.title('Low Frequency'), plt.xticks([]), plt.yticks([])
plt.show()