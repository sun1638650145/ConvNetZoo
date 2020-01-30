import numpy as np
import tensorflow as tf
from modelZoo.AbstractModel import AbstractModel

class AlexNet(AbstractModel):
    """AlexNet单GPU实现"""

    def __init__(self, imgs, num_of_classes):
        self.imgs = imgs
        self.num_of_classes = num_of_classes
        self.parameters = []

        self.conv_layers()
        self.fc_layers()

        self.pred = tf.nn.softmax(self.fc8)

    def conv_layers(self):
        """5个卷积层设计"""

        # conv1
        self.conv1 = self.conv('conv1', self.imgs, 96, 11, 4)
        self.lrn1 = self.lrn('lrn1', self.conv1)
        self.pool1 = self.maxpool('pool1', self.lrn1, 3, padding='VALID')


        # conv1_split
        self.conv1_split = self.split('conv1_split', self.pool1)


        # conv2_up和conv2_down是自相连
        self.conv2_up = self.conv('conv2_up', self.conv1_split[0], 128, 5)
        self.lrn2_up = self.lrn('lrn2_up', self.conv2_up)
        self.pool2_up = self.maxpool('pool2_up', self.lrn2_up, 3, padding='VALID')

        self.conv2_down = self.conv('conv2_down', self.conv1_split[1], 128, 5)
        self.lrn2_down = self.lrn('lrn2_down', self.conv2_down)
        self.pool2_down = self.maxpool('pool2_down', self.lrn2_down, 3, padding='VALID')


        # conv3全相连
        self.pool2_concat = self.concat('pool2_concat', [self.pool2_up, self.pool2_down])

        self.conv3 = self.conv('conv3', self.pool2_concat, 384, 3)


        # conv3_split
        self.conv3_split = self.split('conv3_split', self.conv3)

        # conv4_up和conv4_down是自相连
        self.conv4_up = self.conv('conv4_up', self.conv3_split[0], 192, 3)

        self.conv4_down = self.conv('conv4_down', self.conv3_split[1], 192, 3)


        # conv5_up和conv5_down是自相连
        self.conv5_up = self.conv('conv5_up', self.conv4_up, 128, 3)
        self.pool5_up = self.maxpool('pool5_up', self.conv5_up, 3, padding='VALID')

        self.conv5_down = self.conv('conv5_down', self.conv4_down, 128, 3)
        self.pool5_down = self.maxpool('pool5_down', self.conv5_down, 3, padding='VALID')

        self.pool5_concat = self.concat('pool5_concat', [self.pool5_up, self.pool5_down])

    def fc_layers(self):
        """3个全连接层设计"""

        self.fc6 = self.fc('fc1', self.pool5_concat, 4096)
        self.fc7 = self.fc('fc2', self.fc6, 4096)
        self.fc8 = self.fc('fc3', self.fc7, self.num_of_classes, trainable=True, is_output_layer=True)

    def loadweights(self, weight_file, sess):
        """加载模型"""

        # numpy.load以二进制读出
        weights = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

        # 按照keys升序排序
        keys = sorted(weights.keys())

        count = 0
        for i, j in enumerate(keys):
            if i in [0, 2, 5, 6]:
                # 将i对应的weights和biases赋给perameters
                sess.run(self.parameters[count].assign(np.asarray(weights[j][0])))
                sess.run(self.parameters[count + 1].assign(np.asarray(weights[j][1])))
                count += 2
            elif i in [1, 3, 4]:
                # 将i对应的weights和biases赋给perameters

                """将weights和biases分别拆分"""
                weights_split = np.split(np.asarray(weights[j][0]), 2, axis=3)
                biases_split = np.split(np.asarray(weights[j][1]), 2, axis=0)

                sess.run(self.parameters[count].assign(weights_split[0]))
                sess.run(self.parameters[count + 1].assign(biases_split[0]))
                count += 2

                sess.run(self.parameters[count].assign(weights_split[1]))
                sess.run(self.parameters[count + 1].assign(biases_split[1]))
                count += 2

        print('----------weights loaded----------')