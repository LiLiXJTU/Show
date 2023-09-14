import glob
import warnings

from model.dim2.GFNET_34 import GFNET_34
from model.dim2.U_no_bo import FilterTransU_5_no_Bottleneck
from model.dim2.U_no_trans import FilterTransU_5_notrans

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
from model.dim2.Baseline import BaseLine
import torch.functional as F
import numpy as np
import requests
import torchvision
from model.dim2.FilterTransU import FilterTransU_5
from model.dim2.UNet_ori import UNet
from PIL import Image
from grad_utils import show_cam_on_image, preprocess_image,GradCAM
import cv2
from torchvision import models, transforms
from PIL import ImageFilter
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
import numpy as np


def change_image_channels(image):
    # 3通道转单通道
    if image.mode == 'RGB':
        r, g, b = image.split()
    return r, g, b


def cam_grad(img_name,model_name,img_path):
    image_name = img_path[34:-4]
    # image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
    # image = np.array(Image.open(image_url).convert('RGB'))
    # rgb_img = np.float32(image) / 255

    #task_name='isic_seg'
    #img_path = r"C:\Users\25041\Desktop\NSCLC_img\3.png"
    #img_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\dataset\s001 (67).png'
    # image = Image.open(img_path).convert('RGB').resize((512,512))   #图片尺寸224
    # image = Image.open(img_path).resize((512, 512))  # 图片尺寸224
    image=Image.fromarray(cv2.imread(img_path)).resize((224,224))
    rgb_img = np.float32(image) / 255



    t = transforms.Compose([transforms.Resize((224, 224)),  #输入尺寸224
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=0.1345, std=0.1991)
                            ])

    tensor_img = t(image).unsqueeze(0)

    # model = deeplabv3_resnet50(pretrained=True, progress=False)

    # model =get_model('MSNet_base')
    # load_pretained('MSNet_base',model,'')

    # model =get_model(model_name)
    if model_name=='GFNET_34':
        model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\NSCLC\best_model-GFNET_34.pth.tar'
        model = GFNET_34(3, 32)
    elif model_name=='FilterTransU_5':
        model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\NSCLC\best_model-FilterTransU_5.pth.tar'
        model = FilterTransU_5(3,32)
    elif model_name== 'Baseline':
        model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\NSCLC\best_model-BaseLine.pth.tar'
        model = BaseLine(3,32)
    elif model_name == 'FilterTransU_5_notrans':
        model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\NSCLC\best_model-FilterTransU_5_notrans.pth.tar'
        model = FilterTransU_5_notrans(3, 32)
    elif model_name == 'FilterTransU_5_no_Bottleneck':
        model_path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\pth\NSCLC\best_model-FilterTransU_5_no_Bottleneck.pth.tar'
        model = FilterTransU_5_no_Bottleneck(3, 32)


    # load_pretained(model_name,model,'')
    #
    # # model =get_model('MSNet_2')
    # # load_pretained('MSNet_2',model,'')
    #
    # model = model.eval()

    model.eval()
    model_weights = torch.load(model_path, map_location='cuda')
    model.load_state_dict(model_weights['state_dict'])


    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = tensor_img.cuda()


    class SegmentationModelOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super(SegmentationModelOutputWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x) #返回是元祖需要处理


    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)
    # print(output.shape)
    #多分类
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    #二分类
    #normalized_masks = torch.nn.functional.sigmoid(output).cpu()
    # 此处添加类名
    sem_classes = [
         'ich'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    print(sem_class_to_idx)

    # 将需要进行CAM的类名写至此处
    plaque_category = sem_class_to_idx["ich"]
    #多分类
    plaque_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    #二分类
    #plaque_mask = normalized_masks[0, :, :, :].detach().cpu().numpy()
    plaque_mask[plaque_mask>0.5]=1
    plaque_mask[plaque_mask <= 0.5] = 0
    plaque_mask_uint8 = 255 * np.uint8(plaque_mask == plaque_category)
    plaque_mask_float = np.float32(plaque_mask == plaque_category)

    both_images = np.hstack((image, np.repeat(plaque_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)


    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()

        def __call__(self, model_output):
            return (model_output[self.category, :, :] * self.mask).sum()

    # 此处修改希望得到Grad-CAM图所在的网络层
    # cam_image=''
    target_layers = [model.model.up2]
    targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
    # if model_name=='UNet':
    #     target_layers = [model.model.up3]
    #     targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]

    # elif model_name == 'GFNET_34':
    #     target_layers = [model.model.outc]
    #     targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
    #
    # elif model_name=='MSNet_base' or model_name=='MSNet_1' or model_name=='MSNet_2':
    #     target_layers = [model.model.output1]
    #     targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
    #
    # elif model_name=='UNetplusplus_1' :
    #     target_layers = [model.model.final]
    #     targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
    #
    # elif model_name=='MSMCNet' or model_name=='MSMCNet_base' or model_name=='MSMCNet_sub' or model_name=='MSMCNet_context' :
    #     target_layers = [model.model.final]
    #     targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]

    # else:
    #     print('please input correct model name')

    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image,heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    img = Image.fromarray(cam_image)

    # isic_seg = isic_seg[:, :, 0] + isic_seg[:, :, 1] + isic_seg[:, :, 2]
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    # isic_seg.show()
    path = r'D:\code\CBIM-Medical-Image-Segmentation-main\CBIM-Medical-Image-Segmentation-main\grad_cam/'
    img.save(path + img_name+'/up2/'+image_name + '_' + model_name + ".jpg", qulity=95)

    # r,g,b=change_image_channels(isic_seg)
    # confusion_matrix = pd.DataFrame(b)
    # sns.isic_seg(isic_seg, annot=True, xticklabels=False, yticklabels=False, cmap="Blues")
    # plt.show()

    # img = img.filter(ImageFilter.DETAIL)
    # img = img.filter(ImageFilter.SMOOTH)

    # img.show()
    # 保存位置
    # img.save("./grad_cam/new/" + img_name +'_' + model_name  +".jpg",qulity=95)

if __name__ == '__main__':

    # img_name_list = ['1671']
    img_name_list = ['255','202','257','341','468','845','1082']  #isic
    # img_name_list = ['CTseg (1535)', 'CTseg (1625)', 'CTseg (1115)', 'CTseg (1225)', 'CTseg (331)', 'CTseg (1835)','CTseg (1155)']  # CTseg
    # model_list=['MSMCNet','MSMCNet_base','MSMCNet_sub','MSMCNet_context']
    model_list = ['MSMCNet','MSMCNet_base','MSMCNet_sub','MSMCNet_context']
    # layer_list=['outc','output1','output1']
    # for i in range(len(model_list)):
    #     for img in img_name_list:
    #         cam_grad(img,model_list[i])
    img_path = r'D:\data\NSCLCnew3\Test_Folder\img\*'
    img_list = glob.glob(img_path)
    for img in img_list:
        cam_grad(img_name='NSCLC',model_name='FilterTransU_5_no_Bottleneck',img_path = img) #Baseline  FilterTransU_5 FilterTransU_5_no_Bottleneck FilterTransU_5_notrans



