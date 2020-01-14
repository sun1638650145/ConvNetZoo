import tensorflow as tf

class AbstractModel:
    """定义卷积神经网络的抽象模型"""

    def __init__(self, imgs):
        self.imgs = imgs
        self.parameters = []

    def conv(self, name, input_data, out_channel, trainable=False, use_cudnn_on_gpu=True):
        """卷积方法"""

        # 返回input_data的维度元组的值
        in_channel = input_data.get_shape()[-1]

        with tf.variable_scope(name):
            # 权重和偏置
            # Height Width Input_channel(输入特征) 卷积核数(输出通道数)
            kernel = tf.get_variable('weights', [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable('biases', [out_channel], dtype=tf.float32, trainable=trainable)

            # 卷积操作
            conv_res = tf.nn.conv2d(input_data, kernel, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=use_cudnn_on_gpu)

            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)

        self.parameters += [kernel, biases]

        return out

    def fc(self, name, input_data, out_channel, trainable=False):
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
            weights = tf.get_variable(name='wieghts', shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name='biases', shape=[out_channel], dtype=tf.float32, trainable=trainable)

            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.bias_add(res, biases)
            out = tf.nn.relu(out)

        self.parameters += [weights, biases]

        return out

    def maxpool(self, name, input_data):
        """最大池化方法"""

        # 池化核描述和滑动步长
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)

        return out

    def avgpool(self, name, input_data):
        """均值池化方法"""

        out = tf.nn.avg_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)

        return out

    def saver(self):
        """保存模型"""

        return tf.train.Saver()
