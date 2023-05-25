"""
保存模型
损失函数，欧氏距离
"""

import torch 
# 引入 评价矩阵accuracy
from torchmetrics.functional import accuracy
#引入mseLoss
mse_loss = torch.nn.MSELoss(reduction="sum") 
if __name__ == '__main__':
    pass
    y = torch.tensor([[[0,0],
                       [0,1],
                       [0,2]],

                       [[1,0],
                        [1,1],
                        [1,2]]],dtype=torch.float32)
    print(y.shape)
    a = 9+ 9 +9 +16 +16 +16
    y_pred = torch.tensor([[[0,3],
                       [0,4],
                       [0,5]],
                       [[1,4],
                       [1,5],
                       [1,6]]],dtype=torch.float32)
    
    print(loss_fn(y_pred,y))
    print(f'mse_loss:{mse_loss(y_pred,y)}')
    print(f'accuracy:{accuracy(y_pred,y)}')
