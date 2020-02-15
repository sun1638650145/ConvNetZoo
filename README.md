# ConvNetZoo(一个迁移学习卷积模型库)

​		现如今，众多依靠计算机视觉实现的技术都离不开对预训练的卷积神经网络模型的使用。笔者入行机器学习一年多，每次对图像分类、风格迁移以及目标检测等问题需要重新复现神经网络模型深感头疼，于是乎，开发了这个卷积神经网络库，将陆续支持主流的开源神经网络模型（例如AlexNet(已支持)、VGGNet(已支持)、InceptionNet、ResNet、MobileNet等）

## 项目结构

### 1.modelZoo

​		modelZoo使用TensorFLow复现了多种神经网络模型，定义了一个AbstractModel类，所有的卷积神经网络都继承于此

### 2.utils

​		utils存放了图片预处理的脚本PreProcess.py、将标签转换为独热编码的OneHotEncoder.py、还有读取批次数据的GetBatch.py

### 3.tools

​		tools存放了模型部署和测试的工具脚本，详情见tools/README.MD

## 修复BUG

### 1.Version1.0.1-OneHotEncoder.py
​		读取数据时，类别数大于2且数据不平衡，编码错误的BUG

### 2.Version1.1-GetBatch.py

​		修复macOS读取数据时可能出现的BUG

## 项目更新

### Version1.1 

1. 支持单GPU实现的AlexNet

2. 修复BUG，优化逻辑

### Version1.2

1. 增加固化计算图的脚本tools/Freeze.py
2. 增加运行测试的脚本tools/predict.py
3. 增加一个部署在网页的栗子

## 预训练模型下载链接

1. [VGG16](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)
2. [AlexNet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)

