# python: 3.9+
# encoding: utf-8
# 导入numpy模块并命名为np
import numpy as np  # 导入NumPy库用于高效数值计算
import matplotlib.pyplot as plt
import sys

class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """
        初始化受限玻尔兹曼机（RBM）模型参数

        Args:
            n_hidden (int): 隐藏层单元数量（默认 2）
            n_observe (int): 可见层单元数量（默认 784，如 MNIST 图像 28x28）

        Raises:
            ValueError: 若输入参数非正整数则抛出异常
        """
        # 参数验证：确保隐藏层和可见层单元数量为正整数
        if not (isinstance(n_hidden, int) and n_hidden > 0):
            raise ValueError("隐藏层单元数量 n_hidden 必须为正整数")
        if not (isinstance(n_observe, int) and n_observe > 0):
            raise ValueError("可见层单元数量 n_observe 必须为正整数")
            
        # 初始化模型参数
        self.n_hidden = n_hidden
        self.n_observe = n_observe
        
        # 使用 Xavier/Glorot 初始化权重
        init_std = np.sqrt(2.0 / (self.n_observe + self.n_hidden))
        self.W = np.random.normal(0, init_std, size=(self.n_observe, self.n_hidden))
        
        # 初始化偏置
        self.b_h = np.zeros(self.n_hidden)  # 隐藏层偏置
        self.b_v = np.zeros(self.n_observe)  # 可见层偏置

    def _sigmoid(self, x):
        """Sigmoid激活函数，用于将输入映射到概率空间"""
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        """伯努利采样：根据给定概率生成0或1（用于模拟神经元激活）"""
        return np.random.binomial(1, probs)
    
    def train(self, data, learning_rate=0.1, epochs=10, batch_size=100, k=1):
        """
        使用Contrastive Divergence算法对模型进行训练
        
        参数说明：
        data (numpy.ndarray): 训练数据，形状为 (n_samples, n_observe)
        learning_rate (float): 学习率
        epochs (int): 训练轮数
        batch_size (int): 批处理大小
        k (int): CD-k算法中的k值，即Gibbs采样步数
        """
        # 将数据展平为二维数组 [n_samples, n_observe]
        data_flat = data.reshape(data.shape[0], -1)
        n_samples = data_flat.shape[0]
        
        for epoch in range(epochs):
            # 打乱数据顺序
            np.random.shuffle(data_flat)
            epoch_error = 0.0
            
            # 使用小批量梯度下降法
            for i in range(0, n_samples, batch_size):
                # 获取当前批次的数据
                batch = data_flat[i:i + batch_size]
                batch_size_actual = batch.shape[0]  # 实际批次大小（最后一批可能不同）
                
                # 正相传播
                v0 = batch.astype(np.float64)
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h)
                h0_sample = self._sample_binary(h0_prob)
                
                # 负相传播 (CD-k)
                v_current = v0.copy()
                h_current = h0_sample.copy()
                
                for _ in range(k):
                    h_prob = self._sigmoid(np.dot(v_current, self.W) + self.b_h)
                    h_sample = self._sample_binary(h_prob)
                    v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
                    v_current = self._sample_binary(v_prob)
                
                h1_prob = self._sigmoid(np.dot(v_current, self.W) + self.b_h)
                
                # 计算梯度
                dW = np.dot(v0.T, h0_prob) - np.dot(v_current.T, h1_prob)
                db_v = np.sum(v0 - v_current, axis=0)
                db_h = np.sum(h0_prob - h1_prob, axis=0)
                
                # 更新参数
                self.W += learning_rate * dW / batch_size_actual
                self.b_v += learning_rate * db_v / batch_size_actual
                self.b_h += learning_rate * db_h / batch_size_actual
                
                # 计算重构误差
                batch_error = np.mean((v0 - v_current) ** 2)
                epoch_error += batch_error * batch_size_actual
            
            # 打印本轮的平均重构误差
            print(f"Epoch {epoch+1}/{epochs}, 重构误差: {epoch_error/n_samples:.6f}")
    
    def sample(self, n_samples=1, gibbs_steps=1000):
        """
        从训练好的模型中采样生成新数据（Gibbs采样）
        
        参数:
        n_samples (int): 生成样本数量
        gibbs_steps (int): Gibbs采样步数
        
        返回:
        numpy.ndarray: 生成的样本，形状为 (n_samples, 28, 28)
        """
        samples = []
        
        for _ in range(n_samples):
            # 初始化可见层
            v = np.random.binomial(1, 0.5, self.n_observe)
            
            # 进行Gibbs采样迭代
            for _ in range(gibbs_steps):
                h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
                h_sample = self._sample_binary(h_prob)
                v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
                v = self._sample_binary(v_prob)
            
            # 将最终的可见层向量重塑为28×28的图像格式
            samples.append(v.reshape(28, 28))
        
        return np.array(samples)
    
    def reconstruct(self, data, gibbs_steps=1):
        """
        重构输入数据
        
        参数:
        data (numpy.ndarray): 输入数据，形状为 (n_samples, 28, 28)
        gibbs_steps (int): Gibbs采样步数
        
        返回:
        numpy.ndarray: 重构的数据，形状为 (n_samples, 28, 28)
        """
        data_flat = data.reshape(data.shape[0], -1)
        reconstructions = []
        
        for v in data_flat:
            # 进行Gibbs采样迭代
            for _ in range(gibbs_steps):
                h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
                h_sample = self._sample_binary(h_prob)
                v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
                v = v_prob  # 使用概率而非采样值，得到更平滑的重构
            
            # 将重构的可见层向量重塑为28×28的图像格式
            reconstructions.append(v.reshape(28, 28))
        
        return np.array(reconstructions)

# 使用 MNIST 数据集训练 RBM 模型
if __name__ == '__main__':
    try:
        # 加载二值化的MNIST数据，形状为 (60000, 28, 28)
        mnist = np.load('mnist_bin.npy')  # 60000x28x28
    except IOError:
        print("无法加载MNIST数据文件，请确保mnist_bin.npy文件在正确的路径下")
        sys.exit(1)
    
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols  # 计算单张图片展开后的长度
    print(f"MNIST数据集形状: {mnist.shape}")

    # 初始化RBM对象：100个隐藏节点，784个可见节点（28×28图像）
    # 增加隐藏节点数量可以提高模型表达能力
    rbm = RBM(100, img_size)
    
    # 使用MNIST数据进行训练（增加CD步数提高训练质量）
    rbm.train(mnist, learning_rate=0.1, epochs=20, batch_size=100, k=5)

    # 从模型中采样生成新图像
    generated_images = rbm.sample(n_samples=9)
    
    # 重构一些测试图像
    test_indices = np.random.choice(len(mnist), 9, replace=False)
    test_images = mnist[test_indices]
    reconstructed_images = rbm.reconstruct(test_images)
    
    # 可视化生成的图像
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.suptitle('生成的MNIST数字', fontsize=16)
    plt.show()
    
    # 可视化重构结果
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.axis('off')
    plt.suptitle('重构的MNIST数字', fontsize=16)
    plt.show()
