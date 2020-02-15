"""
    现在支持将CKPT固化成Frozen Model(pb)、TensorFlow SavedModel
    以及直接生成TensorFlow.js的推理模型
"""
import os
import sys
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.saved_model import simple_save
from tensorflow.saved_model import builder
from tensorflow.saved_model import signature_def_utils
from tensorflow.saved_model import tag_constants
# 加入上层目录
sys.path.append('..')
from modelZoo import AlexNet
from modelZoo import VGG16

"""路径信息"""
CKPT_DIR = input("INFO:请输入检查点文件的保存路径(使用绝对路径):")
FROZEN_MODEL_DIR = './frozen_model.pb'
SAVED_MODEL_DIR = './saved_model'

"""超参数"""
try:
    num_of_classes = int(input("INFO:请输入类别总数:"))
except:
    print("\nERROR:格式错误!")
    sys.exit(1)

"""神经网络参数"""
size = int(input("INFO:请输入神经网络的输入尺寸:"))
x = tf.placeholder(tf.float32, shape=[None, size, size, 3], name='x')

"""定义模型"""
model_name = input("INFO:请输入使用的模型:")
if model_name == 'ALexNet':
    cmodel = AlexNet(x, num_of_classes)
    predict = cmodel.pred
elif model_name == 'VGG16':
    cmodel = VGG16(x, num_of_classes)
    predict = cmodel.pred

"""定义一个Saver复现计算图"""
saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.latest_checkpoint(CKPT_DIR)
saver.restore(sess, ckpt)

option = input("INFO:请输入要生成的数据类型(FROZEN_MODEL、SAVED_MODEL或tensorflowjs):")
if option == 'FROZEN_MODEL':
    # 生成FROZEN_MODEL
    output_graph = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['Softmax'])
    with tf.gfile.FastGFile(FROZEN_MODEL_DIR, mode='wb') as f:
        f.write(output_graph.SerializeToString())
elif option == 'SAVED_MODEL':
    # 生成SAVED_MODEL
    default_mode = 'simple'
    if default_mode == 'simple':
        simple_save(sess, SAVED_MODEL_DIR, inputs={'x': x}, outputs={'Softmax': predict})
    elif default_mode == 'complex':
        builder = builder.SavedModelBuilder(SAVED_MODEL_DIR)
        signature = signature_def_utils.predict_signature_def(inputs={'x': x}, outputs={'Softmax': predict})
        builder.add_meta_graph_and_variables(sess, tags=tag_constants.SERVING, signature_def_map={'predict': signature})
        builder.save()
elif option == 'tensorflowjs':
    # 生成tensorflowjs
    simple_save(sess, SAVED_MODEL_DIR, inputs={'x': x}, outputs={'Softmax': predict})
    os.system('sh Converter.sh')

sess.close()
print("INFO:计算图固化成功")