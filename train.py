"""
训练模型
"""

import torch 
from data_setup import create_dataloaders
from model import Resnet50
from utils import loss_fn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from engine import train



loss_fn = loss_fn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnet50(196).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
train_dataloader,test_dataloader = create_dataloaders('data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt',
                                                        'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt',
                                                        batch_size=32,
                                                        num_workers=4,
                                                        transform=ToTensor())
#创建logs文件夹
Path('./logs').mkdir(exist_ok=True,parents=True)
writer = SummaryWriter('./logs')
train(writer,
        100,
        model,
        train_dataloader,
        test_dataloader,
        loss_fn,optimizer,
        device,
        scheduler=scheduler,
        checkpoint_path=Path(r'D:\Repo\faceDetection\model\checkpoint_1.pth'))