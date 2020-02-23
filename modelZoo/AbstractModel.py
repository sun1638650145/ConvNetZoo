import tensorflow as tf

class AbstractModel:
    """定义卷积神经网络的抽象模型"""

    def __init__(self, imgs):
        self.imgs = imgs
        self.parameters = []

    def conv(self, name, input_data, out_channel, kernel_size=3, stride=1, padding='SAME', trainable=False, use_cudnn_on_gpu=False, use_parameters=True):
        """卷积方法（部分早期架构的显卡不支持cudnn加速）"""

        # 返回input_data的维度元组的值
        in_channel = input_data.get_shape()[-1]

        with tf.variable_scope(name):
            # 权重和偏置
            # Height Width Input_channel(输入特征) 卷积核数(输出通道数)
            # 为了兼容Inception模型，卷积层使用了Inception的命名规则
            kernel = tf.get_variable('weights', [kernel_size, kernel_size, in_channel, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable('BatchNorm/beta', [out_channel], dtype=tf.float32, trainable=trainable)
            # 卷积操作
            conv_res = tf.nn.conv2d(input_data, kernel, strides=[1, stride, stride, 1], padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu)

            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)

        if use_parameters == True:
            self.parameters += [kernel, biases]

        return out

    def fc(self, name, input_data, out_channel, trainable=False, is_output_layer=False, use_parameters=True):
        """全连接方法"""

        # 返回input_data的维度元组的值并转换成一个列表
        shape = input_data.get_shape().as_list()
        # RGB三通道图像，后三个维度为size
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        # 灰度图像
        else:
            size = shape[1]
        # -1展平input_data
        input_data_flat = tf.reshape(input_data, [-1, size])

        with tf.variable_scope(name):
            # 权重和偏置
            weights = tf.get_variable(name='weights', shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name='biases', shape=[out_channel], dtype=tf.float32, trainable=trainable)

            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.bias_add(res, biases)

            if is_output_layer == False:
                out = tf.nn.relu(out)

        if use_parameters == True:
            self.parameters += [weights, biases]

        return out

    def maxpool(self, name, input_data, kernel_size=2, stride=2, padding='SAME'):
        """最大池化方法"""

        # 池化核描述和滑动步长
        out = tf.nn.max_pool(input_data, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding=padding, name=name)

        return out

    def avgpool(self, name, input_data, kernel_size=2, stride=2, padding='SAME'):
        """均值池化方法"""

        out = tf.nn.avg_pool(input_data, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding=padding, name=name)

        return out

    def lrn(self, name, input_data):
        """
            局部响应归一化层
            更多细节 http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
        """

        out = tf.nn.lrn(input=input_data, depth_radius=5, bias=1, alpha=1, beta=0.5, name=name)

        return out

    def split(self, name, input_data, num_or_size_splits=2):
        """拆分一个神经网络层"""

        out = tf.split(input_data, num_or_size_splits=num_or_size_splits, axis=3, name=name)

        return out

    def concat(self, name, input_data):
        """拆分两个神经网络层"""

        out = tf.concat(input_data, axis=3, name=name)

        return out

    def dropout(self, name, input_data, keep_prob=0.5):
        """dropout方法"""

        out = tf.nn.dropout(input_data, keep_prob=keep_prob, name=name)

        return out

    def saver(self):
        """保存模型"""

        return tf.train.Saver()
