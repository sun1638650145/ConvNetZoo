import os
import utils
import tensorflow as tf
from modelZoo import VGG16
from time import time

"""计算训练时"""
STARTTIME = time()
"""指定GPU"""
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""路径信息"""
LOG_DIR = 'log/'
CKPT_DIR = 'ckpt/'
TRAIN_DATASETS_DIR = 'train/'
VGG_INITIAL_WEIGHTS = 'vgg16_weights.npz'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

"""超参数"""
batch_size = 32
capacity = 32
learning_rate = 1e-3
train_epochs = 1000
epoch_num = tf.Variable(1, name='epoch_num', trainable=False)
display_step = 10
num_of_classes = 2

"""神经网络参数"""
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
y = tf.placeholder(tf.int32, shape=[None, num_of_classes], name='y')

"""数据加载"""
# 得到图像和标签列表
images_list, labels_list = utils.get_file(TRAIN_DATASETS_DIR)
# 得到图片和标签的训练批次
image_batch, label_batch = utils.get_batch(images_list, labels_list, 224, 224, batch_size, capacity)

"""定义模型"""
cmodel = VGG16(x, num_of_classes)
predict = cmodel.pred

"""评估函数"""
with tf.name_scope('EvaluationFunction'):
    # 按照行取预测值
    correct_prection = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    # 转换为float32
    accuracy = tf.reduce_mean(tf.cast(correct_prection, tf.float32))

"""定义损失和优化器"""
with tf.name_scope('LossandOptimizer'):
    # 使用交叉熵
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits= predict))
    # 使用梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

"""记录tensorboard"""
image_shaped_input = tf.reshape(x, [-1, 224, 224, 3])
tf.summary.image('input', image_shaped_input)
tf.summary.scalar('loss', loss_function)
tf.summary.scalar('accuracy', accuracy)
tf.summary.histogram('predict', predict)
merged_summary_op = tf.summary.merge_all()

"""定义一个saver保存模型数据"""
saver = tf.train.Saver()

sess = tf.Session()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

"""执行训练"""
with tf.name_scope('Train'):
    # 加载初始权重
    cmodel.loadweights(VGG_INITIAL_WEIGHTS, sess)

    # 如果有断点加载断点
    ckpt = tf.train.latest_checkpoint(CKPT_DIR)
    if ckpt != None:
        saver.restore(sess, ckpt)

    # 读取轮数
    start = sess.run(epoch_num)

    # 加入线程协调器
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # 计算运行时间
    epoch_start_time = time()

    for epoch in range(start, train_epochs):
        images, labels = sess.run([image_batch, label_batch])
        labels = utils.onehot_encoder(labels, num_of_classes)

        _, summary_str, loss, acc = sess.run([optimizer, merged_summary_op, loss_function, accuracy], feed_dict={x: images, y: labels})

        # 将summary_str写入tensorboard
        writer.add_summary(summary_str, epoch + 1)

        # 根据显示粒度显示和保存数据
        if epoch % display_step == 0:

            # 计算平均轮数用时
            epoch_end_time = time()
            print('轮数epoch:', epoch+1, '损失loss:', loss, '准确率acc:', acc, '平均epoch用时:', (epoch_end_time - epoch_start_time) / epoch)

            # 保存模型
            saver.save(sess, os.path.join(CKPT_DIR, 'epoch{:06d}.ckpt'.format(epoch+1)))
            # 将epoch_num增加
            sess.run(epoch_num.assign(epoch + 1))
            print('model saved!')

    duration = time() - STARTTIME
    print('训练总用时:', duration)

    coord.request_stop()
    coord.join(threads)

sess.close()
