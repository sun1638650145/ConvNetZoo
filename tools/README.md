# Tools使用教程

​		Tools目前含有一个对训练的模型固化的Freeze.py和进行推理的Predict.py，在终端运行，按照提示输入必要的路径信息，即可自动完成。其中Predict.py的核心处理代码在PredictEngine.py中实现

## 1.Freeze.py

​		Freeze.py支持将CKPT固化成Frozen Model(pb)、TensorFlow SavedModel以及tensorflowjs三种格式，由于现仅支持AlexNet、VGG16两种卷积网络，所以目前仅支持固化这两种模型

## 2.Predict.py

​		Predict.py支持使用CKPT、Frozen Model、SavedModel三种格式的数据进行推理

