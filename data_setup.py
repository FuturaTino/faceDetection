"""
txt数据构成
对数据进行预处理，并加载dataloader，进行批次训练
前196列（0-195）：这些是人脸98个特征点的坐标，按照x1, y1, x2, y2, ..., x98, y98的顺序排列。对于人脸特征点识别任务，我们通常关注的就是这部分数据。

紧接着的4列（196-199）：这是人脸框的坐标，按照x, y, w, h的顺序，其中x, y是左上角的坐标，w和h是人脸框的宽度和高度。

再接下来的6列（200-205）：这些是人脸的属性标签，按照pose（姿态）、expression（表情）、illumination（光照）、make-up（化妆）、occlusion（遮挡）、blur（模糊）的顺序排列。每个属性都用0（不存在）或1（存在）来表示。

最后一列（206）：这是图像的路径。


函数
data_transforms()

get_train_dataset()
get_test_dataset()

get_train_dataloader()
get_test_dataloader()


"""
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os
from pathlib import Path
class WFLWDataset(Dataset):
    """
    """
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.txt_file_dir = Path.joinpath(self.root_dir,txt_file)
        self.annotations = pd.read_csv(self.txt_file_dir,sep=' ',header=None)

        self.transform = transform
    
    def __len__ (self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = Path.joinpath(self.root_dir,'data',self.annotations.iloc[index,-1])
        image = Image.open(img_path)
        landmarks = self.annotations.iloc[index,:-1]
        landmarks = torch.tensor(landmarks,dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        
        return (image, landmarks)


if __name__ == "__main__":
    root = os.getcwd()
    txt_file = 'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt'
    dataset = WFLWDataset(txt_file,root)
    import matplotlib.pyplot as plt
    plt.imshow(dataset[3][0])
    plt.show()