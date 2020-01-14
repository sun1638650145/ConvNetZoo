import os
import numpy as np
import tensorflow as tf
from utils.PreProcess import preprocessed_for_train

def get_file(file_dir):
    """读取文件"""

    images = []
    classes = []
    # os.walk返回的是root(文件夹本身地址)sub_dirs(文件夹下目录地址)files(文件夹下文件地址)
    for root, sub_dirs, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_dirs:
            classes.append(os.path.join(root, name))

    # 将所有的标签
    labels = []
    for one_dir in classes:
        img_number = len(os.listdir(one_dir))
        letter = one_dir.split('/')[-1]
        if letter == 'cat':
            labels = np.append(labels, img_number * [0])
        else:
            labels = np.append(labels, img_number * [1])

    # 打乱顺序，要保证images和labels数量相同，否则会出先temp无法转置
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 将矩阵转换成列表
    images_list = list(temp[:, 0])
    labels_list = list(temp[:, 1])
    labels_list = [int(float(i)) for i in labels_list]

    return images_list, labels_list

def get_batch(images_list, labels_list, height=224, witdh=224, batch_size=32, capacity=32):
    """得到训练的批次"""

    # 使用tf.cast()转换类型
    images = tf.cast(images_list, tf.string)
    labels = tf.cast(labels_list, tf.int32)

    # 创建文件名队列
    input_queue = tf.train.slice_input_producer([images, labels])

    # 读出文件名队列的内容
    images_contents = tf.read_file(input_queue[0])
    labels_contents = input_queue[1]

    # 确保图片编码和通道数一致
    images_decoded = tf.image.decode_jpeg(images_contents, channels=3)
    images_preprocessed = preprocessed_for_train(images_decoded, height, witdh)

    # capacity是队列容量
    images_batch, labels_batch = tf.train.batch([images_preprocessed, labels_contents], batch_size=batch_size, capacity=capacity)

    # 将返回的tensor转换成list
    labels_batch = tf.reshape(labels_batch, [batch_size])

    return images_batch, labels_batch
