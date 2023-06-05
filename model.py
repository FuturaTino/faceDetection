"""
创建模型
# 人脸节点检测一般用什么算法？
# a:
"""


import torch
from torch import nn 
import torchvision 
from torchinfo import summary
# 引入Resnet50
class Resnet50(nn.Module):
    def __init__(self:int,output_size:int)->None:
        super().__init__()
        self.weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.auto_transforms = self.weights.transforms()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.avgpool.parameters():
            param.requires_grad = True
        self.model.fc = nn.Linear(2048, output_size)

    def info(self):
        summary(model=self.model, 
                input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
                # col_names=["input_size"], # uncomment for smaller output
                col_names=["input_size", "output_size", "num_params", "trainable",],
                col_width=20,
                row_settings=["var_names"])
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ =='__main__':
    model = Resnet50(196)
    print(list(model.children()))
    # x = torch.randn(1,3,224,224)
    # y = model(x)
    # print(y.shape)
    # print(y)
    # y = y.reshape(1,-1,2)
    # print(y)
    # y = y[0].detach().numpy()
    # import matplotlib.pyplot as plt
    # plt.scatter(y[:,0],y[:,1])
    # plt.show()
    model.info()
    