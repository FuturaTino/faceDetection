"""
保存模型
损失函数，欧氏距离
"""

import torch 

 
def loss_fn(y_pred:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    """
    损失函数，所有预测点与实际点的欧氏距离 平均值
    """
    return torch.sqrt(torch.sum((y_pred-y)**2,dim=1)).mean()



if __name__ == '__main__':
    pass
    y = torch.tensor([[0,0],[1,0],[2,0]])
    y_pred = torch.tensor([[0,1],[1,1],[2,4]])
    
    print(loss_fn(y_pred,y))