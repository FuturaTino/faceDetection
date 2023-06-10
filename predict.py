import torch


"""
导入模型，去预测一张图片，并画出预测结果在图片上
"""
from model import Resnet50
from PIL import Image
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision import models
# 为gradio创建predict函数



if __name__ =='__main__':

    # 加载模型
    model_path = 'model\RestNet50_45epoch_no_ice.pth'
    cp = torch.load(model_path)
    # print(cp.keys())
    model = Resnet50(196)
    model.load_state_dict(cp['model_state_dict'])
    
    # 读取图片，并预处理将其转成tensor
    from pathlib import Path
    img_path = Path('D:/Repo/faceDetection/gwl.jpg')
    # img = Image.open(img_path)
    auto_transforms = models.ResNet50_Weights.DEFAULT.transforms()
    img = auto_transforms(Image.open(img_path))
    img = img.unsqueeze(0)
    print(f'img shape: {img.shape}')
    
    # 预测
    output = model(img)
    print(f'output shape: {output.shape}')
    output = output.reshape(-1,2)
    output = output.squeeze()
    print(f'output shape: {output.shape}')
    # 将img和output画在一起 用plt
    output = output.detach().numpy()

    plt.imshow(img.squeeze().permute(1,2,0))
    print(f'output shape: {output.shape}')
    plt.scatter(output[:,0],output[:,1]-22)
    plt.savefig('predict_result_gwl.png')
    plt.show()