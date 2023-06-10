import torch
from pathlib import Path
from data_setup import WFLWDataset
from model import Resnet50
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
def predict_plot(model_path:str,test_dir:str,random_seed:int=42):
    """
        调用模型，对WFLW测试集进行预测，通过肉眼，判断模型好坏。预测9张图片，将landmark和图片用plt展示
    """
    model_path = Path(model_path)
    test_dir = Path(test_dir)
    # 加载模型
    model =Resnet50(196)
    cp = torch.load(model_path)
    model.load_state_dict(cp['model_state_dict'])
    # 加载数据集
    dataset = WFLWDataset(test_dir)

    # 随机挑选九张图片，放进list,并预处理成模型输入的格式

    transform = ToTensor()
    random.seed(random_seed)
    num_list = random.sample(range(0,len(dataset)),9)
    img_list =list(map(lambda x:dataset[x][0],num_list))
    tensor_list = list(map(lambda x:transform(x).unsqueeze(0),img_list))  # tensor[1,3,224,224]

    #对九张图进行 预测、绘制
    fig,axes = plt.subplots(3,3)
    for idx,input in enumerate(tensor_list):
        # 预测
        y = model(input).reshape(-1,2)
        y = y.detach().numpy()
        # 绘制
        axes[idx//3,idx%3].imshow(img_list[idx])
        axes[idx//3,idx%3].scatter(y[:,0],y[:,1],s=1)

        for i in range(0,9):
            axes[i//3,i%3].axis('off')
    # 画布长宽扩大一倍
    fig.set_size_inches(18.5, 10.5)
    
    plt.show()


if __name__ == "__main__":
    model_path = 'D:\Repo\\faceDetection\model\RestNet50_18epoch_no_ice.pth'
    test_dir =  'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt'
    predict_plot(model_path=model_path,test_dir=test_dir)


