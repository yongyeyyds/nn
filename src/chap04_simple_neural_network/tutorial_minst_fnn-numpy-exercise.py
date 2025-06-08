#!/usr/bin/env python
# coding: utf-8
# ## 准备数据
# In[1]:

# 导入操作系统接口模块，提供与操作系统交互的功能
import os
# 导入 NumPy 数值计算库，用于高效处理多维数组和矩阵运算
import numpy as np
# 导入 TensorFlow 深度学习框架
import tensorflow as tf
from tqdm import tqdm
# 从 TensorFlow 中导入 Keras 高级 API
from tensorflow import keras
# 从 Keras 中导入常用模块
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# 定义了一个函数 mnist_dataset()，用于加载并预处理 MNIST 数据集
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize 归一化：将像素值从 [0, 255] 缩放到 [0, 1] 范围内
    x = x / 255.0
    x_test = x_test / 255.0

    # 返回处理后的训练集和测试集
    return (x, y), (x_test, y_test)


# ## Demo numpy based auto differentiation
# In[3]:


# 定义矩阵乘法层
class Matmul:
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):
        # 前向传播：执行矩阵乘法，计算 h = x @ W
        h = np.matmul(x, W)
        # 缓存输入 x 和 权重 W，以便在反向传播中计算梯度
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        # 反向传播计算 x 和 W 的梯度
        x = self.mem['x']
        W = self.mem['W']

        '''计算矩阵乘法的对应的梯度'''
        grad_x = np.matmul(grad_y, W.T)# 计算输入x的梯度：将输出梯度grad_y通过权重矩阵W的转置进行反向传播
        grad_W = np.matmul(x.T, grad_y)  # 执行矩形乘法运算，计算梯度

        return grad_x, grad_W


# 定义 ReLU 激活层
class Relu:
    def __init__(self):
        self.mem = {}
        # 初始化记忆字典，用于存储前向传播的输入

    def forward(self, x):
        # 保存输入 x，供反向传播使用
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))
    # ReLU激活函数：x>0时输出x，否则输出0

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算 relu 激活函数对应的梯度'''
        x = self.mem['x']
        grad_x = grad_y * (x > 0)  # ReLU的梯度是1（x>0）或0（x<=0）
        ####################
        return grad_x


# 定义 Softmax 层（输出概率）
class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        # 初始化类实例的基础参数和状态容器
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        # 对输入数据应用指数函数，确保所有值为正
        x_exp = np.exp(x)
        # 计算每个样本的归一化分母（分区函数）
        partition = np.sum(x_exp, axis=1, keepdims=True)
        # 计算 softmax 输出：指数值 / 分区函数
        # 添加 epsilon 防止除零错误（数值稳定性）
        out = x_exp / (partition + self.epsilon)

        # 将计算结果存入内存字典，用于反向传播
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        # 对 grad_y 进行维度扩展
        # 假设 grad_y 是一个形状为 (N, c) 的梯度张量
        # np.expand_dims(grad_y, axis=1) 将其形状变为 (N, 1, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)  # (N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s
        return tmp


# 定义 Log 层（计算 log softmax ，用于交叉熵）
class Log:
    '''
    对最后一个维度执行对数运算
    用于对数似然计算，通常与Softmax结合使用
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y


# ## Gradient check

# In[5]:

# 梯度验证函数
def gradient_check():
    # 验证 Matmul 层
    print("验证 Matmul 层梯度计算:")
    x = np.random.normal(size=[5, 6])
    W = np.random.normal(size=[6, 4])
    matmul = Matmul()
    out = matmul.forward(x, W)  # shape(5, 4)
    grad_x, grad_W = matmul.backward(np.ones_like(out))
    
    with tf.GradientTape() as tape:
        x_tf, W_tf = tf.constant(x), tf.constant(W)
        tape.watch(x_tf)
        tape.watch(W_tf)
        y_tf = tf.matmul(x_tf, W_tf)
        loss_tf = tf.reduce_sum(y_tf)
        grads_tf = tape.gradient(loss_tf, [x_tf, W_tf])
    
    print("手动计算的 grad_x 与 TensorFlow 计算的是否一致:", np.allclose(grad_x, grads_tf[0].numpy()))
    print("手动计算的 grad_W 与 TensorFlow 计算的是否一致:", np.allclose(grad_W, grads_tf[1].numpy()))
    
    # 验证 Relu 层
    print("\n验证 Relu 层梯度计算:")
    x = np.random.normal(size=[5, 6])
    relu = Relu()
    out = relu.forward(x)
    grad_x = relu.backward(np.ones_like(out))
    
    with tf.GradientTape() as tape:
        x_tf = tf.constant(x)
        tape.watch(x_tf)
        y_tf = tf.nn.relu(x_tf)
        loss_tf = tf.reduce_sum(y_tf)
        grads_tf = tape.gradient(loss_tf, x_tf)
    
    print("手动计算的 grad_x 与 TensorFlow 计算的是否一致:", np.allclose(grad_x, grads_tf.numpy()))
    
    # 验证 Softmax + Log 层（等价于 TensorFlow 的 softmax_cross_entropy_with_logits）
    print("\n验证 Softmax + Log 层梯度计算:")
    x = np.random.normal(size=[5, 6], scale=5.0, loc=1)
    label = np.zeros_like(x)
    label[0, 1] = 1.
    label[1, 0] = 1
    label[1, 1] = 1
    label[2, 3] = 1
    label[3, 5] = 1
    label[4, 0] = 1
    
    softmax = Softmax()
    log = Log()
    
    # 前向传播
    soft_out = softmax.forward(x)
    log_out = log.forward(soft_out)
    
    # 反向传播
    log_grad = log.backward(-label)  # 注意这里的负号，对应交叉熵损失的梯度
    soft_grad = softmax.backward(log_grad)
    
    with tf.GradientTape() as tape:
        x_tf = tf.constant(x)
        tape.watch(x_tf)
        # 使用 TensorFlow 的 softmax_cross_entropy_with_logits
        loss_tf = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=x_tf, labels=label))
        grads_tf = tape.gradient(loss_tf, x_tf)
    
    print("手动计算的梯度与 TensorFlow 计算的是否一致:", np.allclose(soft_grad, grads_tf.numpy()))


# 最终梯度验证
# In[6]:

def final_gradient_check():
    x = np.random.normal(size=[5, 6])   # 示例：生成 5 个样本，每个样本 6 维特征
    label = np.zeros_like(x)            # 创建了一个与 x 形状相同的全零标签矩阵
    label[0, 1] = 1.
    label[1, 0] = 1
    label[2, 3] = 1
    label[3, 5] = 1
    label[4, 0] = 1

    W1 = np.random.normal(size=[6, 5])  # 第一层权重 (6→5)
    W2 = np.random.normal(size=[5, 6])  # 第二层权重 (5→6)

    mul_h1 = Matmul()                   # 第一层矩阵乘法
    mul_h2 = Matmul()                   # 第二层矩阵乘法
    relu = Relu()                       # ReLU 激活函数
    softmax = Softmax()
    log = Log()                         # 对数函数
    
    # 手动实现的前向传播过程：
    h1 = mul_h1.forward(x, W1)  # shape(5, 5)
    h1_relu = relu.forward(h1)
    h2 = mul_h2.forward(h1_relu, W2)  # shape(5, 6)
    h2_soft = softmax.forward(h2)
    h2_log = log.forward(h2_soft)
    
    # 手动实现的反向传播过程（计算梯度）：
    h2_log_grad = log.backward(-label)
    h2_soft_grad = softmax.backward(h2_log_grad)
    h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
    h1_relu_grad = relu.backward(h2_grad)
    h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)

    # 使用 TensorFlow 自动微分验证梯度
    with tf.GradientTape() as tape:
        # 将数据转换为 TensorFlow 常量
        x_tf, W1_tf, W2_tf, label_tf = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
        tape.watch(W1_tf)        # 追踪 W1 的梯度
        tape.watch(W2_tf)        # 追踪 W2 的梯度
        h1_tf = tf.matmul(x_tf, W1_tf)
        h1_relu_tf = tf.nn.relu(h1_tf)
        h2_tf = tf.matmul(h1_relu_tf, W2_tf)
        # 使用 TensorFlow 的 softmax_cross_entropy_with_logits 直接计算损失
        loss_tf = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h2_tf, labels=label_tf))
        grads_tf = tape.gradient(loss_tf, [W1_tf, W2_tf])
    
    print("\n最终梯度验证 - W1 梯度是否一致:", np.allclose(W1_grad, grads_tf[0].numpy()))
    print("最终梯度验证 - W2 梯度是否一致:", np.allclose(W2_grad, grads_tf[1].numpy()))


# ## 建立模型

# In[10]:

class myModel:
    def __init__(self):# 初始化模型参数，使用随机正态分布初始化权重矩阵
        # 输入层到隐藏层，增加偏置项
        # W1: 连接输入层(784+1)和隐藏层(100)的权重矩阵
        self.W1 = np.random.normal(size=[28 * 28 + 1, 100], scale=0.01)  
        # 隐藏层到输出层
        # W2: 连接隐藏层(100)和输出层(10)的权重矩阵
        self.W2 = np.random.normal(size=[100, 10], scale=0.01)           
        # 初始化各层操作对象
        self.mul_h1 = Matmul()      # 第一个矩阵乘法层(输入到隐藏层)
        self.mul_h2 = Matmul()      # 第二个矩阵乘法层(隐藏层到输出层)
        self.relu = Relu()          # ReLU激活函数层
        self.softmax = Softmax()    # Softmax激活函数层，用于分类
        self.log = Log()            # 对数运算层，用于计算对数概率

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)                 # 展平图像
        bias = np.ones(shape=[x.shape[0], 1])      # 添加偏置项
        x = np.concatenate([x, bias], axis=1)      # 将偏置向量添加到输入数据中
        
        # 第一层计算：输入层 -> 隐藏层
        self.h1 = self.mul_h1.forward(x, self.W1)  # shape(5, 4),#线性变换
        self.h1_relu = self.relu.forward(self.h1)  # 应用ReLU激活函数
        # 第二层计算：隐藏层 -> 输出层
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)  # 线性变换
        self.h2_soft = self.softmax.forward(self.h2)          # 应用Softmax函数，得到概率分布
        self.h2_log = self.log.forward(self.h2_soft)          # 计算对数概率
        return self.h2_log

    def backward(self, label):
        # 第二层梯度计算
        self.h2_log_grad = self.log.backward(-label)                         # 对数层梯度
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad)          # Softmax层梯度
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad) # 计算W2梯度
        # 第一层梯度计算
        self.h1_relu_grad = self.relu.backward(self.h2_grad)                 # ReLU层梯度
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad) # 计算W1梯度


# ## 计算 loss

# In[11]:

def compute_loss(log_prob, labels):
    return np.mean(np.sum(-log_prob * labels, axis=1))


def compute_accuracy(log_prob, labels):
    predictions = np.argmax(log_prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predictions == truth)


# 单步训练函数
def train_one_step(model, x, y, learning_rate=1e-5):
    # 前向传播：计算模型的输出
    model.forward(x)
    # 反向传播：计算梯度
    model.backward(y)
    # 使用梯度下降法更新权重
    model.W1 -= learning_rate * model.W1_grad
    model.W2 -= learning_rate * model.W2_grad
    # 计算损失值
    loss = compute_loss(model.h2_log, y)
    # 计算准确率
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy


# 测试函数
def test(model, x, y):
    model.forward(x)
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy


# ## 实际训练

# In[12]:

def prepare_data():
    train_data, test_data = mnist_dataset()
    train_label = np.zeros(shape=[train_data[0].shape[0], 10])
    test_label = np.zeros(shape=[test_data[0].shape[0], 10])
    train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1.
    test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1.
    return train_data[0], train_label, test_data[0], test_label


def train(model, train_data, train_label, test_data, test_label, epochs=50, batch_size=128, learning_rate=1e-5):
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []
    num_samples = train_data.shape[0]

    for epoch in tqdm(range(epochs), desc="Training"):
        # 打乱数据顺序
        indices = np.random.permutation(num_samples)
        shuffled_data = train_data[indices]
        shuffled_labels = train_label[indices]
        epoch_loss = 0
        epoch_accuracy = 0

        # 小批量训练
        for i in range(0, num_samples, batch_size):
            # 获取当前批次数据
            batch_data = shuffled_data[i:i + batch_size]
            batch_labels = shuffled_labels[i:i + batch_size]
            # 执行单步训练
            loss, accuracy = train_one_step(model, batch_data, batch_labels, learning_rate)
            # 累计统计量
            epoch_loss += loss * batch_data.shape[0]
            epoch_accuracy += accuracy * batch_data.shape[0]

        # 计算整个 epoch 的平均损失和准确率
        epoch_loss /= num_samples
        epoch_accuracy /= num_samples
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        
        # 评估测试集
        test_loss, test_accuracy = test(model, test_data, test_label)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch}: Train Loss {epoch_loss:.4f}; Train Accuracy {epoch_accuracy:.4f}; Test Loss {test_loss:.4f}; Test Accuracy {test_accuracy:.4f}')
    
    return losses, accuracies, test_losses, test_accuracies


if __name__ == "__main__":
    # 运行梯度验证
    gradient_check()
    final_gradient_check()
    
    # 准备数据
    train_data, train_label, test_data, test_label = prepare_data()
    
    # 训练模型
    model = myModel()
    losses, accuracies, test_losses, test_accuracies = train(
        model, train_data, train_label, test_data, test_label, 
        epochs=50, batch_size=128, learning_rate=1e-5
    )
    
    # 最终测试评估
    test_loss, test_accuracy = test(model, test_data, test_label)
    print(f'Final Test Loss {test_loss:.4f}; Final Test Accuracy {test_accuracy:.4f}')
