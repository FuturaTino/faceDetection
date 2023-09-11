# faceDetection
# 数据集
https://wywu.github.io/projects/LAB/WFLW.html

# 实验环境
torch==2.0.0


# To do:

- [x] 创建dataloader
- [x] 对数据进行预处理
- [x] 构建Resnet，迁移学习，微调分类器
- [x] 写engine和train，保存模型
- [x] 用tensorboard跟踪loss变量和accuracy 
- [ ] 利用model去predict自定义图片
- [ ] 用gradio提供预测人脸特征点服务，部署在huggingface上
- [ ] 总结报告

# 效果图
![img](https://github.com/FuturaTino/TyporaImages/raw/main//TyporaImages/wps1.jpg)

# 评测指标

**测试集**
 ![test_set_loss_acc](https://github.com/FuturaTino/TyporaImages/raw/main//TyporaImages/test_set_loss_acc.png)
