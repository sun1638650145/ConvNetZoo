import numpy as np

def onehot_encoder(labels, num_of_classes):
    """
        将标签转换成独热编码
        例如：现有猫狗两种类别，标签为猫0、狗1；那么转换成独热编码就是猫[1,0]，狗[0,1]
        读取数据的时候不一定读到最大的标签
    """

    # 得到样例数和标签数
    n_samples = len(labels)
    n_classes = num_of_classes

    # 生成一个n_samples行n_classes列的元组
    onehot_labels = np.zeros((n_samples, n_classes))

    # np.arange(n_samples)返回一个a0=0,d=1的等差数列
    # 将onehot_labels对应行的位置的第labels个填上1
    onehot_labels[np.arange(n_samples), labels] = 1

    return onehot_labels