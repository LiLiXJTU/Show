import cv2
import numpy as np
import os
from PIL import Image
import SimpleITK as sitk
# 读取原始图像和分割预测的掩码
# image_path = r'D:\new_data\Fed\1234\Synapse_1234\Test_Folder\img\0036-140.nii'
# image_path = r'D:\new_data\Fed\1234\FLARE22Train_1234\Test_Folder\img\0042-52.nii'
image_path = r'D:\new_data\Fed\1234\amoss_1234\Test_Folder\img\0104-60.nii'
sitk_t1 = sitk.ReadImage(image_path)
image = np.squeeze(sitk.GetArrayFromImage(sitk_t1))
image = cv2.resize(image,(224,224))
mask_path = r'C:\code\xin\fedsemi-l\amos_3_full_our\0104-60_gt_new.png'
mask = cv2.imread(mask_path)

# min_value = np.min(image)
# max_value = np.max(image)
#
# # 对图像像素值进行归一化
#
# normalized_image = (image - min_value) / (max_value - min_value) * 255
# print((-160 - min_value) / (max_value - min_value) * 255)
# print((240 - min_value) / (max_value - min_value) * 255)
# 设置窗宽窗位
window_center = 50
window_width = 400
min_value = window_center - window_width / 2
max_value = window_center + window_width / 2

# 将像素值限制在窗宽窗位的范围内
windowed_image = np.clip(image, min_value, max_value)

# 将像素值映射到0-255的范围
windowed_image = ((windowed_image - min_value) / (max_value - min_value)) * 255
# # 转换数据类型为 uint8
image = windowed_image.astype(np.uint8)

# 将掩码转换为与原始图像相同的数据类型
# mask = mask.astype(np.uint8)
# image = image.astype(np.uint8)
# 将掩码叠加在原始图像上
# img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 将图像转换为具有Alpha通道的图像格式
image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
mask_with_alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

# 为掩码设置透明度
alpha = 0.5  # 设置透明度（0-1之间的值）
mask_with_alpha[:, :, 3] = int(alpha * 255)  # 将Alpha通道设置为透明度值

# 合并图像和掩码
overlay = cv2.addWeighted(image_with_alpha, 1, mask_with_alpha, 0, 1)



# overlay = cv2.addWeighted(img_rgb, 0.7, mask, 0.3, 1)


# overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

img = Image.fromarray(overlay)
path = r'C:\code\xin\fedsemi-l\amos_3_full_our\0104-60-CT.png'
# img.save(path)
cv2.imwrite(path,overlay)


