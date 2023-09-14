from matplotlib import pyplot as plt
# -*- coding: utf-8 -*-
import os

from model.dim2.UNet_ori import UNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
Created on Tue Oct 27 09:25:51 2020

@author: LX
"""

# %%特征可视化
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
import matplotlib as mpl
from PIL import Image
from torchvision import models, transforms
import torch
import timm
class SaveConvFeatures():
    def __init__(self, m):  # module to hook
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.data
    def remove(self):
        self.hook.remove()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# t = transforms.Compose([transforms.Resize((224, 224)),  # 128, 128
#                         transforms.ToTensor()
#                         #transforms.Normalize(mean=0.1345, std=0.1991)
#                         ])
# img_file = r"D:\data\ReDWeb-S\ReDWeb-S\Test\depth\87618102_c0a9039c11_z.png"
# img = Image.open(img_file)
# img = t(img).unsqueeze(0)

img_path = r"D:\data\ReDWeb-S\ReDWeb-S\Test\depth\87618102_c0a9039c11_z.png"

image=Image.fromarray(cv2.imread(img_path)).resize((224,224))
# rgb_img = np.float32(image) / 255



t = transforms.Compose([transforms.Resize((224, 224)),  #输入尺寸224
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=0.1345, std=0.1991)
                            ])

img = t(image).unsqueeze(0)
# img = torch.from_numpy(img).reshape(1,3,224,224)
# model = ProbabilisticUnet(input_channels=3, num_classes=1)
# model.eval()
# model_path=r'D:\li\shiyan\rgb_t\unet_de_en_2\Test_session_07.28_09h59\models\best_model-UNet.pth.tar'
# model_weights = torch.load(model_path, map_location='cuda')
# model.load_state_dict(model_weights['state_dict'])
# model = model.two_unet

model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\models\D\Test_session_08.24_09h52\models\best_model-UNet.pth.tar'
model = UNet()
model.eval()
model_weights = torch.load(model_path, map_location='cuda')
model.load_state_dict(model_weights['state_dict'])
hook_ref = SaveConvFeatures(model.down4)

# 加载.npy文件
np_path  = r'D:\code\task2\UCTransNet-main\np/'
up1_array = np.load(np_path+'xup1.npy')
up1_tensor = torch.from_numpy(up1_array)
up2_array = np.load(np_path+'xup2.npy')
up2_tensor = torch.from_numpy(up2_array)
up3_array = np.load(np_path+'xup3.npy')
up3_tensor = torch.from_numpy(up3_array)
up4_array = np.load(np_path+'xup4.npy')
up4_tensor = torch.from_numpy(up4_array)
#
# print(loaded_tensor)
# Output: tensor([1, 2, 3])
with torch.no_grad():
    #model(img,up1_tensor,up2_tensor,up3_tensor,up4_tensor)
    model(img)


conv_features = hook_ref.features  # [1,2048,7,7]
print('特征图输出维度：', conv_features.shape)  # 其实得到特征图之后可以自己编写绘图程序
hook_ref.remove()


# Visualize 64 feature maps from each layer
processed =[]
for feature_map in conv_features:
    #feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.numpy())
# for fm in processed:
#     print(fm.shape)

fig = plt.figure(figsize=(30,50))
for i in range(len(processed)):
    a = fig.add_subplot(5,4,i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")

# cmap1 = copy.copy(mpl.cm.viridis)
# norm1 = mpl.colors.Normalize(vmin=-1., vmax=1.)
# im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
#
# plt.colorbar(im1, cax=None, orientation='horizontal')
# plt.colorbar()
save_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\grad_cam\unet_seg/'
plt.savefig(save_path+'outc.png',bbox_inches = 'tight')