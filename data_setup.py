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
import time
import random 
NUM_WORKERS = os.cpu_count()
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
    
    side = max(x_center-x_min,y_center-y_min)  # 稍微扩大区域，包括更多细节
    side = side * 1.5
    rect = (x_center - side, y_center - side, x_center + side, y_center + side)
    img = img.crop(rect)
    return img,rect

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
    target_size = (224,224)
    # 获取image的宽高
    pts = pts/image.size * target_size[0] ###################
    image = image.resize(target_size,Image.LANCZOS)
    return image,pts

def _fliplr(image:Image,pts:np.ndarray):
    """
    随机水平翻转图片
    Args:
        image: PIL.Image
        pts: 人脸98个特征点的坐标
    return:
        image: PIL.Image
        pts: 反转后，对应的人脸98个特征点的坐标
    """
    a = np.ndarray((98,2),dtype=np.float32)
    if random.random() >=0.5:
        print(f'翻转前的pts \n{pts}')
        pts[:,0] = 224 - pts[:,0]
        pts = pts[_fliplr.perm]
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        print(f'翻转后的pts \n{pts}')
        # 将反转后对应点的坐标赋值给原来的点
        # 以前五个点为例，原来的点为[1,2,3,4,5....]，反转后的点为[32,31,30,29,28....]，详情看
    return image,pts

# 统一图片平均亮度
def _relight(image:Image)->Image:
    r,g,b = ImageStat.Stat(image).mean
    brightness = math.sqrt(0.241*r**2 + 0.691*g**2 + 0.068*b**2)
    # 0.241, 0.691, 0.068是RGB转换为YIQ的转换矩阵
    image = ImageEnhance.Brightness(image).enhance(128/brightness)
    return image


# 裁剪 缩放都已经在上面完成了
def data_transforms()->transforms.Compose:
    """
    数据预处理
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        meta = np.array(self.annotations.iloc[index,:-1],dtype=np.float32)

        rect = [int(x) for x in meta[196:200]]
        pts = meta[0:196].reshape(98,2)
        # 裁剪图片，保留脸部区域
        image,rect = _crop_face(image,rect) # 变成了正方形框
        pts -= rect[0:2]
        image,pts = _resize(image,pts)
        # 随机水平翻转图片
        image,pts = _fliplr(image,pts)
        # 统一图片平均亮度
        image = _relight(image)
        # 转成Tensor
        if self.transform:
            image = self.transform(image)
            # 转为Tensor float32
            pts = torch.from_numpy(pts).float()
        # image (3,224,224)landmarks (98,2)
        return (image, pts)



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


if __name__ == "__main__":
    #全排列向量_perm ，记录关键点反转后对应关系 shape(98,)
    # example:  _fliplr[96] = 97  ,说明第96个关键点反转后对应的关键点是第97个
    # 为什么说是全排列呢？因为这个_perm是在 初始化的时候就固定好了，不会改变的
    random.seed(42)
    _fliplr.perm = np.load('data/fliplr_perm.npy')
    train_txt_file = 'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt'
    test_txt_file = 'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt'
    dataset  = WFLWDataset(train_txt_file,transform=data_transforms()) # (img,landmarks)


    # idx = random.randint(0,len(dataset))

    print(_fliplr.perm)
    idx = 3

    # 这里很重要，dataset调用一次，就会调用一次__getitem__方法，不要写成dataset[idx][0],dataset[idx][1]，这样会调用两次，导致图片和关键点不对应
    # 调试 六个小时，才发现这个问题，哎，学会了debug，hhhhh
    # img,pts = dataset[idx]
    # img = img.permute(1,2,0)
    # pts = pts.numpy()
    # print(type(pts))
    # print(f'最终获取的Pts:\n {pts}')
    # # print(f'_fliplr.perm:\n {_fliplr.perm}')
    # plt.imshow(img)
    # plt.scatter(pts[:,0],pts[:,1],s=10,c='r')
    # plt.show()

    # 一次展示九张图
    fig = plt.figure()
    fig.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)

    for i in range(9):
        idx = random.randint(0,len(dataset))
        img,landmarks = dataset[idx]
        img.permute(1,2,0)
        landmarks = landmarks.numpy()
        ax = fig.add_subplot(3,3,i+1)
        # 去除坐标轴
        ax.axis('off')
        #图片间隔为0

        ax.imshow(img.permute(1,2,0))
        ax.scatter(landmarks[:,0],landmarks[:,1],s=10,c='r')
    plt.show()
