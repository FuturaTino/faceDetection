"""
完成一batch的训练和测试
用tensorboardX记录训练和测试的loss和acc
函数
train_step()
test_step()

"""
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model import Resnet50
from data_setup import create_dataloaders
from torch import nn
import torch
from tqdm.auto import tqdm
from typing import Dict,List,Tuple
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional

def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device,
               k=5)->Tuple[float,float]:
    """
    该函数将目标PyTorch模型设置为训练模式，然后运行所有必要的训练步骤（前向传递、损失计算、优化器步骤）。

    参数：

    model：要训练的PyTorch模型。
    dataloader：用于训练模型的DataLoader实例。
    loss_fn：要最小化的PyTorch损失函数。
    optimizer：用于最小化损失函数的PyTorch优化器。
    device：要计算的目标设备（例如“cuda”或“cpu”）。
    返回值：
    以(train_loss, train_accuracy)形式返回训练损失和训练准确度指标的元组。例如：

    (0.1112, 0.8743)
    """    
    model.train()

    train_loss,train_acc = 0,0
    batch_size = dataloader.batch_size

    for x,y in tqdm(dataloader,desc='Training'):  
        x,y = x.to(device),y.to(device)

        
        #前向传播
        y_pred = model(x)
        y = y.reshape(-1,196)
        # 计算损失函数, 1 batch的平均损失
        loss = loss_fn(y_pred,y)  # 1 batch的平均损失,1 batch时98个点，所以这里loss含义时98个点的总偏移量的平方，当loss小于98,1个点偏移不到1
        train_loss += loss.item()
        # 优化器清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        #优化器更新参数
        optimizer.step()


        # reshape
        y_pred = y_pred.reshape(-1,2)
        y = y.reshape(-1,2)
        # 计算准确率,将每个特征点与真实值进行比较，如果距离小于0.05，认为预测正确
        # 损失函数即为欧式距离
        distances = torch.sqrt(torch.sum((y_pred-y)**2,dim=-1)).to(device) # distances [batch_size*98,]
        acc = torch.sum(distances<k).item() / (batch_size*98)  # 一共batch_size *98个点，其中距离小于k=0.05的频率
        train_acc += acc
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss,train_acc


def test_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device,
               k=5)->Tuple[float,float]:
    """
    该函数将目标PyTorch模型设置为测试模式，运行一次前向传播

    参数：

    model：要训练的PyTorch模型。
    dataloader：用于训练模型的DataLoader实例。
    loss_fn：要最小化的PyTorch损失函数。
    optimizer：用于最小化损失函数的PyTorch优化器。
    device：要计算的目标设备（例如“cuda”或“cpu”）。
    返回值：
    以(train_loss, train_accuracy)形式返回预测损失和训练准确度指标的元组。例如：
    
    (0.1112, 0.8743)
    """    
    model.eval()

    #创建loss, acc
    test_loss,test_acc = 0,0
    # 获取batch_size
    batch_size = dataloader.batch_size
    with torch.no_grad():
        #q: no_grad和model.eval()有什么区别？
        #a: model.eval()是将模型设置为测试模式，不会影响梯度计算，但是会影响BN和dropout的计算
        #q:bn和drop是什么
        #a:bn是batch normalization，批标准化，是一种加速神经网络训练的方法，通过对每一层的输入进行归一化，使得每一层的输入都满足均值为0，方差为1的标准正态分布，从而加速训练
        #a:dropout是一种正则化方法，通过在训练过程中随机让隐藏层的部分神经元失效，从而减少模型的过拟合
        for x,y in tqdm(dataloader,desc="Testing"):
            x,y = x.to(device),y.to(device)
            # 前向传播
            y_pred = model(x)

            y = y.reshape(-1,196)
            # 计算损失函数
            loss = loss_fn(y_pred,y)
            test_loss += loss.item()

            # reshape
            y_pred = y_pred.reshape(-1,2)
            y = y.reshape(-1,2)
            # 计算acc
            distances = torch.sqrt(torch.sum((y_pred-y)**2,dim=-1)).to(device) # [batch_size*98,]
            acc = torch.sum(distances<k).item() / (batch_size*98)
            test_acc += acc
            
    test_loss /= len(dataloader)
    test_acc /=len(dataloader)
    return test_loss,test_acc 


def train(writer:SummaryWriter,
          epochs:int,
          model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer, 
          device:torch.device,
          model_save_path:Path=None,
          scheduler:torch.optim.lr_scheduler.StepLR =None,
          k=5,
          checkpoint_path:Optional[str]=None):
    

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        # 获取模型的参数
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint from {checkpoint_path}')

    Path('./model').mkdir(exist_ok=True,parents=True)
    for epoch in range(start_epoch,epochs):
        train_loss,train_acc = train_step(model,train_dataloader,loss_fn,optimizer,device,k)
        test_loss,test_acc = test_step(model,test_dataloader,loss_fn,optimizer,device,k)

        print(f'epoch:{epoch},train_loss:{train_loss},train_acc:{train_acc},test_loss:{test_loss},test_acc:{test_acc}')


        # q:为什么很少保存整个模型
        # a:因为模型的参数很多，保存整个模型会占用很大的空间，而且很多时候我们只需要模型的参数，而不需要模型的结构
        
        if epoch % 9 == 0:
            # 保存模型、优化器、epoch
            checkpoint = {
                'model_state_dict':model.state_dict(),
                'optimizer':optimizer,
                'epoch':epoch
            }
            torch.save(checkpoint,f'{model_save_path}/checkpoint_{epoch}.pth')
            print(f'save checkpoint to {model_save_path}/checkpoint_{epoch}.pth')
        writer.add_scalar('train_loss',train_loss,epoch)
        writer.add_scalar('train_acc',train_acc,epoch)
        writer.add_scalar('test_loss',test_loss,epoch)
        writer.add_scalar('test_acc',test_acc,epoch)
        writer.flush()

        if scheduler is not None:
            # Step the scheduler
            scheduler.step()
    writer.close()


if __name__ =='__main__':
    a= torch.rand((3,4))
    print(torch.sum(a,dim=-1).shape)
    print(torch.sum(a,dim=-1))




    