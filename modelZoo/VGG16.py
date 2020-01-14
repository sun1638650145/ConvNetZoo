import numpy as np
import tensorflow as tf
from modelZoo.AbstractModel import AbstractModel

class VGG16(AbstractModel):
    """Visual Geometry Group 16"""

    def __init__(self, imgs):
        self.imgs = imgs
        self.parameters = []

        self.conv_layers()
        self.fc_layers()

        # 不初始化完fc_layers没有self.fc8
        self.pred = tf.nn.softmax(self.fc8)

    def conv_layers(self):
        """13个卷积层设计"""

        self.conv1_1 = self.conv('conv1_1', self.imgs, 64, trainable=False, use_cudnn_on_gpu=True)
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, 64, trainable=False, use_cudnn_on_gpu=True)
        self.pool1 = self.maxpool('pool1', self.conv1_2)

        self.conv2_1 = self.conv('conv2_1', self.pool1, 128, trainable=False, use_cudnn_on_gpu=True)
        self.conv2_2 = self.conv('conv2_2', self.conv2_1, 128, trainable=False, use_cudnn_on_gpu=True)
        self.pool2 = self.maxpool('pool2', self.conv2_2)

        self.conv3_1 = self.conv('conv3_1', self.pool2, 256, trainable=False, use_cudnn_on_gpu=True)
        self.conv3_2 = self.conv('conv3_2', self.conv3_1, 256, trainable=False, use_cudnn_on_gpu=True)
        self.conv3_3 = self.conv('conv3_3', self.conv3_2, 256, trainable=False, use_cudnn_on_gpu=True)
        self.pool3 = self.maxpool('pool3', self.conv3_3)

        self.conv4_1 = self.conv('conv4_1', self.pool3, 512, trainable=False, use_cudnn_on_gpu=True)
        self.conv4_2 = self.conv('conv4_2', self.conv4_1, 512, trainable=False, use_cudnn_on_gpu=True)
        self.conv4_3 = self.conv('conv4_3', self.conv4_2, 512, trainable=False, use_cudnn_on_gpu=True)
        self.pool4 = self.maxpool('pool4', self.conv4_3)

        self.conv5_1 = self.conv('conv5_1', self.pool4, 512, trainable=False, use_cudnn_on_gpu=True)
        self.conv5_2 = self.conv('conv5_2', self.conv5_1, 512, trainable=False, use_cudnn_on_gpu=True)
        self.conv5_3 = self.conv('conv5_3', self.conv5_2, 512, trainable=False, use_cudnn_on_gpu=True)
        self.pool5 = self.maxpool('pool5', self.conv5_3)

    def fc_layers(self):
        """3个全连接层设计"""
        self.fc6 = self.fc('fc1', self.pool5, 4096, trainable=False)
        self.fc7 = self.fc('fc2', self.fc6, 4096, trainable=False)
        self.fc8 = self.fc('fc3', self.fc7, 2, trainable=True)

    def loadweights(self, weight_file, sess):
        """加载模型"""

        # numpy.load以二进制读出
        weights = np.load(weight_file)

        # 按照keys升序排序
        keys = sorted(weights.keys())

        for i, j in enumerate(keys):
            if i not in [30, 31]:
                # 将i对应的value赋给perameters
                sess.run(self.parameters[i].assign(weights[j]))
        print('----------weights loaded----------')