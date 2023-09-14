import cv2
import glob
import os
path_to_parent_folder = r'C:\Users\25041\Desktop\MSD_show'
folder_path = glob.glob(path_to_parent_folder+'\\*')
pic_name = 'cropped_3.png'
# mask_name = pic_name.split('.')[0]+'_mask.png'
mask_name = pic_name
for folder in folder_path:
    image_path = os.path.join(r'C:\Users\25041\Desktop\MSD_img',pic_name)
    pre_mask_path = os.path.join(folder,mask_name)
    mask_path = os.path.join(r'C:\Users\25041\Desktop\MSD_mask',pic_name)
    img = cv2.imread(image_path)
    mask_yuan = cv2.imread(mask_path)
    pre_mask = cv2.imread(pre_mask_path)
    mask = cv2.cvtColor(mask_yuan, cv2.COLOR_BGR2GRAY)
    pre_mask = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    ret1, thresh1 = cv2.threshold(pre_mask, 127, 255, 0)

    contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #第一个参数是轮廓
    cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1)

    contours1, im1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #第一个参数是轮廓
    cv2.drawContours(image=img, contours=contours1, contourIdx=-1, color=(0, 0, 255), thickness=1) # 预测是红色
    p = r'C:\Users\25041\Desktop\MSD_pre\3_'
    cv2.imwrite(p+folder[32:]+'.png',img)