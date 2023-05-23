"""
to do

txt数据构成
对数据进行预处理，并加载dataloader，进行批次训练
前196列（0-195）：这些是人脸98个特征点的坐标，按照x1, y1, x2, y2, ..., x98, y98的顺序排列。对于人脸特征点识别任务，我们通常关注的就是这部分数据。

紧接着的4列（196-199）：这是人脸框的坐标，按照x_min, y_min, x_max, y_max的顺序，其中x_min, y_min是左上角的坐标，x_max和y_max是人脸框的右下角坐标

再接下来的6列（200-205）：这些是人脸的属性标签，按照pose（姿态）、expression（表情）、illumination（光照）、make-up（化妆）、occlusion（遮挡）、blur（模糊）的顺序排列。每个属性都用0（不存在）或1（存在）来表示。

最后一列（206）：这是图像的路径。

类
WFLWDataset(Dataset)

函数
data_transforms()

create_dataloaders()

"""

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image,ImageStat,ImageEnhance
import torch
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import math
NUM_WORKERS = os.cpu_count()
def _resize(image:Image,pts):
    """
    将图片resize成224*224
    Args:
        image: PIL.Image
        pts: 人脸98个特征点的坐标，按照x1, y1, x2, y2, ..., x98, y98的顺序排列
    return:
        image: PIL.Image
        pts: 人脸98个特征点的坐标，按照x1, y1, x2, y2, ..., x98, y98的顺序排列
    """
    pts = np.array(pts)
    pts.resize(98,2)
    target_size = (224,224)
    print(pts.shape)
    print(image.size)
    pts = pts/255 * target_size[0]
    image = image.resize(target_size,Image.ANTIALIAS)
    return image,pts
    
# 裁剪 缩放都已经在上面完成了
def data_transforms()->transforms.Compose:
    """
    数据预处理
    """
    return transforms.Compose([
        transforms.ToTensor()
    ])


# 统一图片平均亮度
def _relight(image:Image)->Image:
    r,g,b = ImageStat.Stat(image).mean
    brightness = math.sqrt(0.241*r**2 + 0.691*g**2 + 0.068*b**2)
    # 0.241, 0.691, 0.068是RGB转换为YIQ的转换矩阵
    image = ImageEnhance.Brightness(image).enhance(128/brightness)
    return image
class WFLWDataset(Dataset):
    """
    """
    def __init__(self, txt_file, transform=None):
        self.annotations = pd.read_csv(txt_file,sep=' ',header=None)
        self.transform = transform
    
    def __len__ (self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = Path('data/' + self.annotations.iloc[index,-1])
        image = Image.open(img_path)
        landmarks = np.array(self.annotations.iloc[index,:-1])



        # 裁剪图片，保留脸部区域
        image = _crop_face(image,landmarks[196:200])
        image,landmarks = _resize(image,landmarks[0:196])
        # 统一图片平均亮度
        image = _relight(image)

        if self.transform:
            image = self.transform(image)
        
        return (image, landmarks)



def create_dataloaders(
        train_txt_dir,
        test_txt_dir,
        transform:transforms.Compose,
        batch_size:int,
        num_workers:int = NUM_WORKERS):
    """
    创建 train_loader 和 test_loader
    输入数据集中WFLW_anoations的中txt文件路径，转成pytorch中的dataloader

    Args:
        train_txt_dir: 训练集的txt文件路径
        test_txt_dir: 测试集的txt文件路径
        transform: 数据预处理
        batch_size: 批次大小
        num_workers: dataloader处理数据的线程数

    return:
        train_loader: 训练集的dataloader一个元组 (train_loader, test_loader)
        一条记录就是txt文件中一行，包括图片路径和图片中的98个特征点的坐标，以及人脸框的坐标和人脸属性标签，和照片路径
    """
    train_data = WFLWDataset(train_txt_dir,transform=transform)
    test_data = WFLWDataset(test_txt_dir,transform=transform)
    
    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    # pin_memory=True: 如果设置为True，那么数据加载器将会在返回它们之前，将张量复制到CUDA固定内存中
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return (train_dataloader, test_dataloader)

# 根据照片脸部正方形坐标，裁剪图片，只保留脸部的正方形区域,并resize成224,224
def _crop_face(img,rect):
    """
    根据照片脸部正方形坐标，裁剪图片，只保留脸部的正方形区域,并resize成224,224
    Args:
        img: PIL.Image
        rect: 人脸框的坐标，按照x, y, w, h的顺序，其中x, y是左上角的坐标，w和h是人脸框的宽度和高度。
    return:
        img: PIL.Image
    """
    x_min, y_min, x_max,y_max = rect
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    side = max(x_center-x_min,y_center-y_min) * 1.5 # 稍微扩大区域，包括更多细节
    rect = (x_center - side, y_center - side, x_center + side, y_center + side)
    img = img.crop(rect)
    return img

if __name__ == "__main__":
    train_txt_file = 'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt'
    test_txt_file = 'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt'

    dataset  = WFLWDataset(train_txt_file,transform=data_transforms()) # (img,landmarks)
    print(dataset[0][0].shape)
    plt.imshow(dataset[1][0].permute(1,2,0))
    plt.show()