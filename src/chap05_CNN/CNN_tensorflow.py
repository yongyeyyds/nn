#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 使用input_data.read_data_sets函数加载MNIST数据集，'MNIST_data'是数据集存储的目录路径，one_hot=True表示将标签转换为one-hot编码格式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 超参数设置
learning_rate = 1e-4  # 优化器学习率，控制参数更新步长
keep_prob_rate = 0.7  # Dropout保留概率，防止过拟合
max_epoch = 2000  # 最大训练轮数，每轮处理一个batch的数据


def compute_accuracy(v_xs, v_ys):
    """
    计算模型在给定数据集上的准确率。

    参数:
        v_xs: 输入数据，形状为[batch_size, 784]
        v_ys: 真实标签，形状为[batch_size, 10]

    返回:
        result: 模型预测准确率
    """
    global prediction
    # 获取模型预测结果，keep_prob=1表示测试阶段不使用dropout
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # 比较预测类别与真实类别，得到布尔型数组
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 将布尔型转为浮点数并计算平均值，即准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 执行计算图获取准确率数值
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    """
    初始化权重变量，使用截断正态分布避免梯度消失/爆炸

    参数:
        shape: 权重张量形状，如[filter_height, filter_width, in_channels, out_channels]

    返回:
        tf.Variable: 初始化后的权重变量
    """
    # 使用截断正态分布初始化权重，标准差0.1，超出2倍标准差的值会被重新生成
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 将初始化值转换为可训练的TensorFlow变量
    return tf.Variable(initial)


def bias_variable(shape):
    """
    初始化卷积层/全连接层的偏置变量
    
    参数:
        shape: 偏置的维度（如[32]）
    
    返回:
        tf.Variable: 使用常数0.1初始化的偏置变量（避免死神经元）
    """
    # 使用常数0.1初始化偏置，避免ReLU神经元因初始输出为0而无法激活
    initial = tf.constant(0.1, shape=shape)
    # 创建可训练的TensorFlow变量
    return tf.Variable(initial)


def conv2d(x, W, padding='SAME', strides=[1, 1, 1, 1]):
    """
    实现二维卷积操作，增加了参数灵活性和异常处理
    
    参数:
        x (tf.Tensor): 输入张量，形状为[batch, height, width, channels]
        W (tf.Tensor): 卷积核权重，形状为[filter_height, filter_width, in_channels, out_channels]
        padding (str): 填充方式，'SAME'或'VALID'
        strides (list): 步长列表，[1, stride_h, stride_w, 1]
        
    返回:
        tf.Tensor: 卷积结果
    异常:
        ValueError: 如果 padding 不是 'SAME' 或 'VALID'，会抛出异常。
        TypeError: 如果输入参数类型不正确，会抛出异常。
    """
    # 验证输入类型
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)

    # 检查权重参数W是否为TensorFlow张量
    if not tf.is_tensor(W):
        # 如果不是张量类型，抛出类型错误异常
        # 错误信息包含期望的类型和实际传入的类型
        raise TypeError(f"Expected W to be a tf.Tensor, but got {type(W)}.")

    # 验证卷积操作的padding参数是否合法
    if padding not in ['SAME', 'VALID']:
        # 如果padding不是'SAME'或'VALID'，抛出值错误异常
        # 错误信息显示无效的输入值，并提示有效选项
        raise ValueError(f"Invalid padding value: {padding}. Must be 'SAME' or 'VALID'.")

    # 验证strides参数的格式，应该是一个长度为4的列表
    if len(strides) != 4:
        raise ValueError(f"Strides should be a list of length 4, but got list of length {len(strides)}.")
    
    # 执行卷积操作
    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)
    
    # 添加批归一化以提高训练稳定性（注释掉，根据实际需求启用）
    # conv = tf.layers.batch_normalization(conv, training=is_training)
    
    return conv


def max_pool_2x2(x: tf.Tensor,
    pool_size: int = 2,
    strides: int = 2,
    padding: str = 'SAME',
    data_format: str = 'NHWC'
) -> tf.Tensor:
    """
    实现2x2最大池化操作，减小特征图尺寸，保留主要特征
    
    参数:
        x: 输入张量，NHWC格式为[batch, height, width, channels]
        pool_size: 池化窗口大小
        strides: 步长
        padding: 填充方式
        data_format: 数据格式，NHWC或NCHW
        
    返回:
        tf.Tensor: 池化后的张量
    """
    # 验证参数合法性
    if padding not in ['SAME', 'VALID']:
        raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}.")
    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError(f"data_format must be 'NHWC' or 'NCHW', got {data_format}.")
    
    # 构造池化核和步长参数
    if data_format == 'NHWC':
        ksize = [1, pool_size, pool_size, 1]
        strides = [1, strides, strides, 1]
    else:  # NCHW
        ksize = [1, 1, pool_size, pool_size]
        strides = [1, 1, strides, strides]
    
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)


# 定义网络输入占位符
# xs: MNIST图像数据，形状为[None, 784]（784=28*28）
# ys: 对应标签，形状为[None, 10]（10个类别）
# keep_prob: Dropout保留率，训练时<1，测试时=1
xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 归一化处理，将像素值从[0,255]缩放到[0,1]
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# 将一维输入重塑为四维张量[batch, height, width, channels]
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1表示自动计算batch_size维度


# 第一层卷积+池化
# 卷积层：7x7卷积核，32个输出通道，提取低级特征
# 池化层：2x2最大池化，缩小特征图尺寸为14x14
W_conv1 = weight_variable([7, 7, 1, 32])  # 卷积核形状[7,7,1,32]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出特征图尺寸: 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # 输出特征图尺寸: 14x14x32


# 第二层卷积+池化
# 卷积层：5x5卷积核，64个输出通道，提取中级特征
# 池化层：2x2最大池化，缩小特征图尺寸为7x7
W_conv2 = weight_variable([5, 5, 32, 64])  # 卷积核形状[5,5,32,64]
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出特征图尺寸: 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # 输出特征图尺寸: 7x7x64


# 第一个全连接层
# 将卷积输出展平为一维向量，连接1024个神经元
# 使用ReLU激活函数引入非线性
W_fc1 = weight_variable([7*7*64, 1024])  # 输入维度7*7*64，输出维度1024
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 将7x7x64的特征图展平为一维向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 全连接+ReLU激活

# 应用Dropout防止过拟合
# keep_prob在训练时设为0.7，测试时设为1
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 第二个全连接层（输出层）
# 将1024维特征映射到10个类别
# 使用softmax激活函数输出类别概率分布
W_fc2 = weight_variable([1024, 10])  # 输入维度1024，输出维度10（对应10个数字类别）
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 输出形状[batch_size, 10]


# 定义损失函数和优化器
# 使用交叉熵损失函数衡量预测分布与真实分布的差异
# 交叉熵公式：H(p,q) = -Σ p(x)log(q(x))
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])
)
# 使用Adam优化器最小化损失函数
# Adam优化器结合了Adagrad和RMSProp的优点，自适应调整学习率
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# 训练和评估模型
with tf.Session() as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # 迭代训练max_epoch轮
    for i in range(max_epoch):
        # 从训练集获取一个batch的数据（100个样本）
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 执行一次参数更新
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: keep_prob_rate})
        
        # 每100轮在测试集的前1000个样本上评估模型准确率
        if i % 100 == 0:
            print(f"Epoch {i}, Test accuracy: {compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]):.4f}")
