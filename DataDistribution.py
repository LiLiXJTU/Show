import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.decomposition import PCA
import glob
import cv2
import umap
from sklearn.manifold import TSNE
import tifffile as tiff
# 读取数据
data = []
labels = []
# folder1 = r'D:\new_data\Fed\1234\300slices\FLARE22Train_1234\Test_Folder\img/'
# folder2 = r'D:\new_data\Fed\1234\300slices\TCIA_1234\Test_Folder\img/'
# folder3 = r'D:\new_data\Fed\1234\300slices\amoss_1234\Test_Folder\img/'
# folder4 = r'D:\new_data\Fed\1234\300slices\FLARE22Train_1234\Test_Folder\img/'
# folder5 = r'D:\new_data\Fed\1234\300slices\TCIA_1234\Test_Folder\img/'
# folder6 = r'D:\new_data\Fed\1234\300slices\amoss_1234\Test_Folder\img/'



folder1 = r'/mnt/sda/li/Fed_data/1234/300slices/FLARE22Train_1234/Train_Folder/img/'
folder2 = r'/mnt/sda/li/Fed_data/1234/300slices/TCIA_1234/Train_Folder/img/'
folder3 = r'/mnt/sda/li/Fed_data/1234/300slices/amoss_1234/Train_Folder/img/'
folder4 = r'/mnt/sda/li/Fed_data/1234/300slices/FLARE22Train_1234/Test_Folder/img/'
folder5 = r'/mnt/sda/li/Fed_data/1234/300slices/TCIA_1234/Test_Folder/img/'
folder6 = r'/mnt/sda/li/Fed_data/1234/300slices/amoss_1234/Test_Folder/img/'
folder7 = r'/mnt/sda/li/Fed_data/1234/300slices/Synapse_1234/Test_Folder/img/'
# 循环遍历每个文件并读取像素值

for i, folder in enumerate([folder2, folder5,folder7]):
    files = glob.glob(os.path.join(folder, '*.nii'))
    for j, file in enumerate(files):
        filepath = file
        print(filepath)
        filename = os.path.basename(filepath)
        print(filename)
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)
        array_new = cv2.resize(array, (224, 224))
        if 'Train_Folder' in filepath:
            if 'FLARE22Train_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/FLARE22Train_1234/Train_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\FLARE22Train_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
            elif 'TCIA_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/TCIA_1234/Train_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\TCIA_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
            elif 'amoss_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/amoss_1234/Train_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\amoss_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
        else:
            if 'FLARE22Train_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/FLARE22Train_1234/Test_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\FLARE22Train_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
            elif 'TCIA_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/TCIA_1234/Test_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\TCIA_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
            elif 'amoss_1234' in filepath:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/amoss_1234/Test_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\amoss_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
            else:
                path = r'/mnt/sda/li/Fed_data/1234/300slices/Synapse_1234/Test_Folder/labelcol/'
                #path = r'D:\new_data\Fed\1234\300slices\amoss_1234\Test_Folder\labelcol/'
                maskpath = path+filename[:-3]+'tif'
        tif_mask = tiff.imread(maskpath)
        mask = cv2.resize(tif_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask[mask==1]=1
        mask[mask==2]=1
        mask[mask==3]=1
        mask[mask==4]=1
        array = array_new * mask
        #array = array_new
        if j ==0:
            NEW = array.flatten()
        if j >0:
            NEW = np.vstack((NEW,array.flatten()))
        data.append(np.transpose(array).flatten())
        labels.append(i)

    #for filename in os.listdir(folder):


#数据归一化
# data = np.array(data)
# data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# PCA降维
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(data)
print(reduced_data.shape)

# reducer = umap.UMAP(min_dist=0.1, n_components=3)
# reduced_data = reducer.fit_transform(data)

# tsne = TSNE(n_components=3)
# reduced_data = tsne.fit_transform(data)
#深蓝色，深黄色，桃红色
color = ['#444693', '#f47920','#f05b72','#4e72b8','#fedcbd','#f8aba6']
# 可视化
# for i in range(len(labels)):
#     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c=color[labels[i]],linewidths=0.001)
# # 可视化
# # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
# plt.colorbar()
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.savefig('/mnt/sda/li/pca_FLARE22Train_test_2d')
# plt.show()


#3d
fig = plt.figure()
ax = fig.gca(projection="3d")

for i in range(len(labels)):
    ax.scatter(reduced_data[i, 0], reduced_data[i, 1], zs=reduced_data[i, 2], c=color[labels[i]],linewidths=0.001)

ax.set(xlabel="X", ylabel="Y", zlabel="Z")
plt.savefig('/mnt/sda/li/pca_test_3d_mask_1234')
plt.show()