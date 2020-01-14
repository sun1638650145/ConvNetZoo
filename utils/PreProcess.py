import tensorflow as tf

RESZIE_SIDE_MIN = 256
RESZIE_SIDE_MAX = 512

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

def smallest_size_at_least(height, witdh, smallest_side):
    """
        按照比例将长和宽调整到标准的长度范围（需要较小的一个数满足要求即可）
        例如：（放大）height = 256 witdh = 128 samllest_side = 512
            转换后new_height = 1024 new_witdh = 512
        例如：（缩小）height = 256 witdh = 512 samllest_side = 128
            转换后new_height =128 new_witdh = 256
    """

    # 将输入最小边长转换为tensor
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    # 转换成float32
    height = tf.to_float(height)
    witdh = tf.to_float(witdh)
    smallest_side = tf.to_float(smallest_side)

    # 如果condition H>W，按照W处理；反之，按照H处理
    scale = tf.cond(tf.greater(height, witdh), lambda: smallest_side / witdh, lambda: smallest_side / height)

    new_height = tf.to_int32(height * scale)
    new_witdh = tf.to_int32(witdh * scale)

    return new_height, new_witdh

def aspect_preserving_resize(images, smallest_side):
    """按照原始的长宽比例保存改变尺寸后的图片"""

    #将输入最小边长转换为tensor
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    #得到图像的长宽
    shape = tf.shape(images)
    height = shape[0]
    witdh = shape[1]

    # 按要求调整到标准的输入范围
    new_height, new_witdh = smallest_size_at_least(height, witdh, smallest_side)

    # 给图片添加第0维batch维
    images = tf.expand_dims(images, 0)

    # 使用双线性插值法（不进行中心对齐）
    resized_images = tf.image.resize_bilinear(images, [new_height, new_witdh], align_corners=False)

    # tf.squeeze()去除为1的维度，这个维度没有意义
    # 例如：tensor[[[0,1,2],[3,4,5]]] shape(1,2,3)
    # squeeze()后 tensor[[0,1,2],[3,4,5]] shape(2,3)
    resized_images = tf.squeeze(resized_images)
    resized_images.set_shape([None, None, 3])

    return resized_images

def crop(image, offset_height, offset_witdh, crop_height, crop_witdh):
    """根据偏移量和尺寸进行裁切"""

    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['张量的秩必须等于3'])
    with tf.control_dependencies([rank_assertion]):
        # 拼接
        cropped_shape = tf.stack([crop_height, crop_witdh, original_shape[2]])

    size_assertion = tf.Assert(tf.logical_and(tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_witdh)), ['裁切尺寸大于等于原始尺寸'])
    offsets = tf.to_int32(tf.stack([offset_height, offset_witdh, 0]))

    with tf.control_dependencies([size_assertion]):
        # 裁切图片
        image = tf.slice(image, offsets, cropped_shape)

    return tf.reshape(image, cropped_shape)

def random_crop(images_list, crop_height, crop_witdh):
    """随机裁切给定的图片列表"""

    if not images_list:
        raise ValueError('空图片列表')

    rank_assertions = []
    for i in range(len(images_list)):
        # 返回张量的秩
        image_rank = tf.rank(images_list[i])

        # 断言秩是否是3
        rank_assert = tf.Assert(tf.equal(image_rank, 3), ['张量%s的秩错误[expected][actual]', images_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(images_list[0])

    image_height = image_shape[0]
    image_witdh = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height), tf.greater_equal(image_witdh, crop_witdh)), ['裁切尺寸大于等于原始尺寸'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(images_list)):
        image = images_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        witdh = shape[1]

        height_assert = tf.Assert(tf.equal(height, image_height), ['张量%s的长度错误[expected][actual]', image.name, height, image_height])
        witdh_assert = tf.Assert(tf.equal(witdh, image_witdh), ['张量%s的宽度错误[expected][actual]', image.name, witdh, image_witdh])

        # 在asserts列表后追加多个元素
        asserts.extend([height_assert, witdh_assert])

    with tf.control_dependencies(asserts):
        # 最大偏移长度
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        # 最大偏移宽度
        max_offset_witdh = tf.reshape(image_witdh - crop_witdh + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_witdh = tf.random_uniform([], maxval=max_offset_witdh, dtype=tf.int32)

    return [crop(image, offset_height, offset_witdh, crop_height, crop_witdh) for image in images_list]

def mean_images_subtraction(image, means):
    """对每个色彩通道进行减均值处理"""

    if image.get_shape().ndims != 3:
        raise ValueError('输入的维度必须是3维')

    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('均值的维度和图片的通道数不匹配')

    # 按照channel维将image分成单独的RGB三通道图像
    channels = tf.split(value=image, num_or_size_splits=num_channels, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]

    # 按照channel维将图片拼回
    return tf.concat(values=channels, axis=2)

def preprocessed_for_train(images, output_height, output_witdh, resize_side_min=RESZIE_SIDE_MIN, resize_side_max=RESZIE_SIDE_MAX):
    """
        预处理训练图片，使用了
        随机缩放aspect_preserving_resize
        随机裁切random_crop
        随机翻转random_flip_left_right
        最后进行了减均值处理
    """

    # 产生在[minval,maxval)均匀分布的随机数
    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

    # 按照随机尺寸缩放图片
    images = aspect_preserving_resize(images, resize_side)

    # 随机裁切图片
    images = random_crop([images], output_height, output_witdh)[0]
    images.set_shape([output_height, output_witdh, 3])

    images = tf.to_float(images)
    # 随机左右翻转
    images = tf.image.random_flip_left_right(images)

    return mean_images_subtraction(images, [R_MEAN, G_MEAN, B_MEAN])