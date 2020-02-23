import os
import sys
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.saved_model import tag_constants
sys.path.append("..")
from utils import PreProcess
from modelZoo import AlexNet
from modelZoo import VGG16
from modelZoo import GoogLeNet

def compute_time(function):
    """计算测试时间"""

    def wrapper(*arg, **kwargs):
        start = time()
        result = function(*arg, **kwargs)
        end = time()
        print('测试总用时:%.3fs' % (end - start))
        return result

    return wrapper

@compute_time
def predict_with_ckpt_singlephoto(image, net_size, model_name, num_of_classes, ckpt_dir, predict_list):
    """使用检查点进行预测"""

    # 对输入的图片进行预处理
    img = PreProcess.preprocessed_for_predict(image, net_size)

    # 复现模型
    input_tensor = tf.placeholder(tf.float32, shape=[None, net_size, net_size, 3])
    if model_name == 'AlexNet':
        cmodel = AlexNet(input_tensor, num_of_classes)
        predict = cmodel.pred
    elif model_name == 'VGG16':
        cmodel = VGG16(input_tensor, num_of_classes)
        predict = cmodel.pred
    elif model_name == 'GoogLeNet':
        cmodel = GoogLeNet(input_tensor, num_of_classes)
        predict = cmodel.pred

    # 生成saver
    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, ckpt)

    # 进行预测
    prob = sess.run(predict, feed_dict={input_tensor: [img]})
    index = np.argmax(prob)

    sess.close()

    return prob, predict_list[index]

@compute_time
def predict_with_ckpt(images_dir, net_size, model_name, num_of_classes, ckpt_dir, predict_list):
    """使用检查点进行预测"""

    # 复现模型
    input_tensor = tf.placeholder(tf.float32, shape=[None, net_size, net_size, 3])
    if model_name == 'AlexNet':
        cmodel = AlexNet(input_tensor, num_of_classes)
        predict = cmodel.pred
    elif model_name == 'VGG16':
        cmodel = VGG16(input_tensor, num_of_classes)
        predict = cmodel.pred
    elif model_name == 'GoogLeNet':
        cmodel = GoogLeNet(input_tensor, num_of_classes)
        predict = cmodel.pred

    # 生成saver
    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, ckpt)

    # 进行预测
    prob_list = []
    ans_list = []
    for i in range(len(os.listdir(images_dir))):
        # 对输入的图片进行预处理
        image = images_dir + str(i) + '.jpg'
        img = PreProcess.preprocessed_for_predict(image, net_size)

        prob = sess.run(predict, feed_dict={input_tensor: [img]})
        prob_list.append(prob)
        index = np.argmax(prob)
        ans_list.append(index)

    sess.close()

    return prob_list, ans_list

@compute_time
def predict_with_pb(image, net_size, pb_dir, predict_list):
    """使用Frozen Model进行预测"""

    # 对输入的图片进行预处理
    img = PreProcess.preprocessed_for_predict(image, net_size)

    # 复现模型
    input_tensor = tf.placeholder(tf.float32, shape=[None, net_size, net_size, 3])
    with open(pb_dir, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        predict = tf.import_graph_def(graph_def,
                                   input_map={'x:0': input_tensor},
                                   return_elements=['Softmax:0'])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # 进行预测
    prob = sess.run(predict, feed_dict={input_tensor: [img]})
    index = np.argmax(prob)

    sess.close()

    return prob, predict_list[index]

@compute_time
def predict_with_sm(image, net_size, sm_dir, predict_list):
    """使用SavedModel进行预测"""

    # 对输入的图片进行预处理
    img = PreProcess.preprocessed_for_predict(image, net_size)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # 复现模型
    tf.saved_model.loader.load(sess, tag_constants.SERVING, sm_dir)
    input_tensor = sess.graph.get_tensor_by_name('x:0')
    predict = sess.graph.get_tensor_by_name('Softmax:0')

    # 进行预测
    prob = sess.run(predict, feed_dict={input_tensor: [img]})
    index = np.argmax(prob)

    sess.close()

    return prob, predict_list[index]