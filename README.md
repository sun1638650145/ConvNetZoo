# ConvNetZoo(一个迁移学习卷积模型库)

​		现如今，众多依靠计算机视觉实现的技术都离不开对预训练的卷积神经网络模型的使用。笔者入行机器学习一年多，每次对图像分类、风格迁移以及目标检测等问题需要重新复现神经网络模型深感头疼，于是乎，开发了这个卷积神经网络库，将陆续支持主流的开源神经网络模型（例如AlexNet、VGGNet、InceptionNet、ResNet、MobileNet等）

## 项目结构

### 1.modelZoo

​		modelZoo使用TensorFLow复现了多种神经网络模型，定义了一个AbstractModel类，所有的卷积神经网络都继承于此

### 2.utils

​		utils存放了图片预处理的脚本PreProcess.py、将标签转换为独热编码的OneHotEncoder.py、还有读取批次数据的GetBatch.py

