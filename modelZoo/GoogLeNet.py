import tensorflow as tf
from modelZoo.AbstractModel import AbstractModel

class GoogLeNet(AbstractModel):
    """GoogleNet 没有使用Batch Normalization"""

    def __init__(self, imgs, num_of_classes):
        self.imgs = imgs
        self.num_of_classes = num_of_classes
        self.restore_list = []

        self.layers()
        self.restore_variables()

        self.pred = self.softmax0 * 0.3 + self.softmax1 * 0.3 + self.softmax2 * 0.4

    def InceptionModule(self, name, input_data, previous_shape):
        """Inception module with dimension reductions"""
        with tf.variable_scope(name):
            self.branch1_conv1 = self.conv('Branch_0/Conv2d_0a_1x1', input_data, previous_shape[0], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch2_conv1 = self.conv('Branch_1/Conv2d_0a_1x1', input_data, previous_shape[1], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
            self.branch2_conv2 = self.conv('Branch_1/Conv2d_0b_3x3', self.branch2_conv1, previous_shape[2], kernel_size=3, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch3_conv1 = self.conv('Branch_2/Conv2d_0a_1x1', input_data, previous_shape[3], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
            self.branch3_conv2 = self.conv('Branch_2/Conv2d_0b_3x3', self.branch3_conv1, previous_shape[4], kernel_size=3, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch4_maxpool1 = self.maxpool('branch4_maxpool1', input_data, kernel_size=3, stride=1)
            self.branch4_conv1 = self.conv('Branch_3/Conv2d_0b_1x1', self.branch4_maxpool1, previous_shape[5], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            out = self.concat(name, [self.branch1_conv1, self.branch2_conv2, self.branch3_conv2, self.branch4_conv1])

            return out

    def InceptionModule_error(self, name, input_data, previous_shape):
        """尽管你可能不相信，但确实是有一层的名字写错了"""
        with tf.variable_scope(name):
            self.branch1_conv1 = self.conv('Branch_0/Conv2d_0a_1x1', input_data, previous_shape[0], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch2_conv1 = self.conv('Branch_1/Conv2d_0a_1x1', input_data, previous_shape[1], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
            self.branch2_conv2 = self.conv('Branch_1/Conv2d_0b_3x3', self.branch2_conv1, previous_shape[2], kernel_size=3, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch3_conv1 = self.conv('Branch_2/Conv2d_0a_1x1', input_data, previous_shape[3], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
            self.branch3_conv2 = self.conv('Branch_2/Conv2d_0a_3x3', self.branch3_conv1, previous_shape[4], kernel_size=3, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            self.branch4_maxpool1 = self.maxpool('branch4_maxpool1', input_data, kernel_size=3, stride=1)
            self.branch4_conv1 = self.conv('Branch_3/Conv2d_0b_1x1', self.branch4_maxpool1, previous_shape[5], kernel_size=1, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)

            out = self.concat(name, [self.branch1_conv1, self.branch2_conv2, self.branch3_conv2, self.branch4_conv1])

            return out

    def AuxiliaryModule(self, name, input_data):
        """auxiliary"""
        with tf.variable_scope(name):
            self.auxiliary_avgpool = self.avgpool('avgpool', input_data, kernel_size=5, stride=3, padding='VALID')
            self.auxiliary_conv = self.conv('conv', self.auxiliary_avgpool, 128, kernel_size=1, stride=1, trainable=False, use_cudnn_on_gpu=False, use_parameters=False)
            self.auxiliary_fc1 = self.fc('fc1', self.auxiliary_conv, 1024, use_parameters=False)
            self.auxiliary_dropout = self.dropout('dropout', self.auxiliary_fc1, keep_prob=0.7)
            self.auxiliary_fc2 = self.fc('fc2', self.auxiliary_dropout, self.num_of_classes, trainable=True, is_output_layer=True, use_parameters=False)

            auxiliary_softmax = tf.nn.softmax(self.auxiliary_fc2, name=name)

            return auxiliary_softmax

    def layers(self):

        self.conv1 = self.conv('InceptionV1/Conv2d_1a_7x7', self.imgs, 64, kernel_size=7, stride=2, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
        self.maxpool1 = self.maxpool('maxpool1', self.conv1, kernel_size=3, stride=2)
        self.lrn1 = self.lrn('lrn1', self.maxpool1)

        self.conv2_1 = self.conv('InceptionV1/Conv2d_2b_1x1', self.lrn1, 64, kernel_size=1, stride=1, padding='VALID', trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
        self.conv2_2 = self.conv('InceptionV1/Conv2d_2c_3x3', self.conv2_1, 192, kernel_size=3, stride=1, trainable=True, use_cudnn_on_gpu=False, use_parameters=False)
        self.lrn2 = self.lrn('lrn2', self.conv2_2)
        self.maxpool2 = self.maxpool('maxpool2', self.lrn2, kernel_size=3, stride=2)

        self.inception3a = self.InceptionModule('InceptionV1/Mixed_3b', self.maxpool2, [64, 96, 128, 16, 32, 32])
        self.inception3b = self.InceptionModule('InceptionV1/Mixed_3c', self.inception3a, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = self.maxpool('maxpool3', self.inception3b, kernel_size=3, stride=2)

        self.inception4a = self.InceptionModule('InceptionV1/Mixed_4b', self.maxpool3, [192, 96, 208, 16, 48, 64])
        self.inception4b = self.InceptionModule('InceptionV1/Mixed_4c', self.inception4a, [160, 112, 224, 24, 64, 64])
        self.inception4c = self.InceptionModule('InceptionV1/Mixed_4d', self.inception4b, [128, 128, 256, 24, 64, 64])
        self.inception4d = self.InceptionModule('InceptionV1/Mixed_4e', self.inception4c, [112, 144, 288, 32, 64, 64])
        self.inception4e = self.InceptionModule('InceptionV1/Mixed_4f', self.inception4d, [256, 160, 320, 32, 128, 128])
        self.maxpool4 = self.maxpool('maxpool4', self.inception4e, kernel_size=3, stride=2)

        # auxiliary
        self.softmax0 = self.AuxiliaryModule('auxiliary_softmax0', self.inception4a)
        self.softmax1 = self.AuxiliaryModule('auxiliary_softmax1', self.inception4d)

        self.inception5a = self.InceptionModule_error('InceptionV1/Mixed_5b', self.maxpool4, [256, 160, 320, 32, 128, 128])
        self.inception5b = self.InceptionModule('InceptionV1/Mixed_5c', self.inception5a, [384, 192, 384, 48, 128, 128])
        self.avgpool5 = self.avgpool('avgpool5', self.inception5b, kernel_size=7, stride=1, padding='VALID')
        self.dropout5 = self.dropout('dropout5', self.avgpool5, keep_prob=0.4)

        self.fc6 = self.fc('fc3', self.dropout5, self.num_of_classes, trainable=True, is_output_layer=True, use_parameters=False)
        self.softmax2 = tf.nn.softmax(self.fc6, name='softmax2')

    def restore_variables(self):
        """预训练变量列表"""

        for variable in tf.global_variables():
            if variable.name.startswith('InceptionV1/Conv2d'):
                self.restore_list.append(variable)
            if variable.name.startswith('InceptionV1/Mixed'):
                self.restore_list.append(variable)

    def loadweights(self, ckpt, sess, saver):
        """加载参数"""
        saver.restore(sess, ckpt)
        print('----------weights loaded----------')

#alias
InceptionV1 = GoogLeNet