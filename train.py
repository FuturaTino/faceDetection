"""
训练模型
"""

import torch 
from data_setup import create_dataloaders
from model import Resnet50
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from engine import train
from opt import get_opt




if __name__ == "__main__":
  #用argparse来管理超参数
  args = get_opt()


  #hyperparameters

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Resnet50(196).to(device)
  loss_fn = torch.nn.MSELoss(reduction="mean")
  optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

  train_dir = args.train_dir
  test_dir = args.test_dir
  epochs = args.epochs
  batch_size = args.batch_size
  model_save_path = args.model_save_path



  train_dataloader,test_dataloader = create_dataloaders(train_dir,
                                                        test_dir,
                                                          batch_size=batch_size,
                                                          num_workers=4,
                                                          transform=ToTensor())
  #创建logs文件夹
  Path('./logs').mkdir(exist_ok=True,parents=True)
  writer = SummaryWriter('./logs')
  train(writer,
          epochs,
          model,
          train_dataloader,
          test_dataloader,
          loss_fn,optimizer,
          device,
          model_save_path=model_save_path,
          scheduler=scheduler,
          checkpoint_path=None)