# !/usr/bin/env python
# coding: utf-8

# # 诗歌生成
# # 数据处理

# In[1]:
# 导入了多个用于构建和训练深度学习模型的Python库和模块
import numpy as np  # 导入NumPy库，用于高性能科学计算和多维数组处理 常用功能：数组操作、数学函数、线性代数等
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets

# 定义特殊标记：开始标记和结束标记
start_token = 'bos'  # Beginning of sentence
end_token = 'eos'    # End of sentence

def process_dataset(fileName):
    """处理诗歌数据集，构建词汇表和数字索引的诗歌数据
    
    Args:
        fileName: 诗歌文本文件路径，格式为"标题:内容"
        
    Returns:
        instances: 数字索引化的诗歌列表，每个诗歌是数字id的列表
        word2id: 词语到数字id的映射字典
        id2word: 数字id到词语的映射字典
    """
    examples = []  # 存储处理后的诗歌样本
    
    # 读取文件并处理每行诗歌
    with open(fileName, 'r', encoding='utf-8') as fd:
        for line in fd:
            # 分割标题和内容
            outs = line.strip().split(':')
            content = ''.join(outs[1:])  # 取内容部分
            
            # 构建序列：[开始标记] + 内容字符列表 + [结束标记]
            ins = [start_token] + list(content) + [end_token]
            
            # 过滤掉长度过长的样本，避免训练困难
            if len(ins) > 200:
                continue
                
            examples.append(ins)
            
    # 统计词频，用于构建词汇表
    counter = collections.Counter()
    for e in examples:
        for w in e:
            counter[w] += 1
    
    # 按词频从高到低排序
    sorted_counter = sorted(counter.items(), key=lambda x: -x[1])
    
    # 构建词汇表：添加PAD(填充)和UNK(未知词)标记
    words, _ = zip(*sorted_counter)  # 对tuple进行解压，得到words列表代表所有字符
    words = ('PAD', 'UNK') + words[:len(words)]  # 扩展词汇表：在原始词汇表前添加特殊标记
    
    # 创建词语到id的映射
    word2id = dict(zip(words, range(len(words))))
    
    # 创建id到词语的映射
    id2word = {word2id[k]:k for k in word2id}
    
    # 将诗歌转换为数字id序列
    indexed_examples = [[word2id[w] for w in poem] for poem in examples]
    
    # 记录每个诗歌的长度，用于后续批次处理
    seqlen = [len(e) for e in indexed_examples]
    
    # 组合诗歌数据和对应长度
    instances = list(zip(indexed_examples, seqlen))
    
    return instances, word2id, id2word

def poem_dataset():
    """创建诗歌数据集的TensorFlow Dataset对象
    
    Returns:
        ds: 处理好的tf.data.Dataset对象
        word2id: 词语到id的映射
        id2word: id到词语的映射
    """
    # 处理原始数据
    instances, word2id, id2word = process_dataset('../poems.txt')
    
    # 创建Dataset，使用生成器生成数据
    ds = tf.data.Dataset.from_generator(
        lambda: [ins for ins in instances], 
        (tf.int64, tf.int64),  # 输出类型：诗歌序列和长度
        (tf.TensorShape([None]), tf.TensorShape([])))  # 输出形状
    
    # 数据预处理
    ds = ds.shuffle(buffer_size=10240)  # 打乱数据，提高训练稳定性
    
    # 批处理并填充到相同长度，便于并行计算
    ds = ds.padded_batch(100, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    
    # 为语言模型准备输入输出对：输入是前n-1个词，输出是后n-1个词
    # 例如：输入 "床前明月光"，输出 "前明月光eos"
    ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen-1))
    
    return ds, word2id, id2word


# # 模型代码

# In[2]:


class myRNNModel(keras.Model):
    """基于RNN的诗歌生成模型"""
    def __init__(self, w2id):
        """初始化模型
        
        Args:
            w2id: 词语到id的映射字典，用于确定词汇表大小
        """
        super().__init__()
        self.v_sz = len(w2id)  # 词汇表大小
        
        # 嵌入层：将词语id映射为密集向量，便于模型处理
        self.embed_layer = tf.keras.layers.Embedding(
            self.v_sz, 64,  # 64维嵌入向量
            batch_input_shape=[None, None])  # 支持变长序列
        
        # RNN单元：使用SimpleRNNCell，是最基础的RNN实现
        self.rnncell = tf.keras.layers.SimpleRNNCell(128)  # 128维隐藏状态
        
        # RNN层：包装RNN单元，处理序列输入
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        
        # 输出层：预测下一个词的概率分布
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
    @tf.function
    def call(self, inp_ids):
        """模型前向传播
        
        Args:
            inp_ids: 输入词id序列，形状(batch_size, seq_len)
            
        Returns:
            logits: 每个位置的下一个词预测，形状(batch_size, seq_len, vocab_size)
        """
        # 1. 词嵌入层：将词id转换为密集向量
        x = self.embed_layer(inp_ids)  # (batch_size, seq_len, emb_size)
        
        # 2. RNN层：处理序列，捕获上下文信息
        rnn_out = self.rnn_layer(x)  # (batch_size, seq_len, hidden_size)
        
        # 3. 输出层：预测下一个词
        logits = self.dense(rnn_out)  # (batch_size, seq_len, vocab_size)
        
        return logits
        
    # 定义了一个用于文本生成的函数 get_next_token，基于当前词和RNN的隐藏状态预测下一个词并更新状态
    @tf.function
    def get_next_token(self, x, state):
        """生成下一个词（用于文本生成阶段）
        
        Args:
            x: 当前词id，形状(batch_size,)
            state: RNN的隐藏状态
            
        Returns:
            out: 预测的下一个词id
            state: 更新后的RNN状态
        """
        # 1. 嵌入当前词
        inp_emb = self.embed_layer(x)  # shape(batch_size, emb_size)
        
        # 2. RNN前向传播，更新隐藏状态
        h, state = self.rnncell(inp_emb, state)  # h: (batch_size, hidden_size)
        
        # 3. 预测下一个词
        logits = self.dense(h)  # (batch_size, vocab_size)
        
        # 4. 选择概率最高的词作为预测结果
        out = tf.argmax(logits, axis=-1)
        
        return out, state


# ## 辅助函数：计算序列损失

# In[3]:


def mkMask(input_tensor, maxLen):
    """创建掩码，用于处理变长序列
    
    Args:
        input_tensor: 包含序列长度的张量
        maxLen: 最大序列长度
        
    Returns:
        与input_tensor形状相同的布尔掩码，有效位置为True，填充位置为False
    """

    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])
    
    # 使用tf.reshape将input_tensor展平为一维张量oneDtensor
    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    
    # 使用tf.sequence_mask函数生成一个掩码张量flat_mask
    # 序列实际长度内的位置为True，超出长度的位置为False
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    
    return tf.reshape(flat_mask, shape_of_output)

def reduce_avg(reduce_target, lengths, dim):
    """沿指定维度计算掩码后的平均值（忽略填充部分）
    
    Args:
        reduce_target: 需要求平均的张量
        lengths: 每个序列的实际长度
        dim: 沿哪个维度求平均
        
    Returns:
        掩码后的平均值
    """
    # 获取长度张量和目标张量的形状
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    
    # 验证输入张量的维度是否符合要求
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    
    if len(shape_of_target) < dim+1:
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    # 计算目标张量与长度张量的秩差
    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    
    # 获取目标维度的最大长度，并生成掩码矩阵
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    
    # 根据秩差调整掩码和长度张量的形状，以便广播
    if rank_diff != 0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    
    # 重塑张量以匹配目标张量的形状
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    # 将掩码应用到目标张量上，无效位置（填充部分）值为0
    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    # 在指定维度上求和（不保留归约后的维度）
    red_sum = tf.reduce_sum(mask_target, axis=[dim], keepdims=False)
    
    # 计算平均值：总和 / 有效元素数量 + 极小值（防止除以零）
    red_avg = red_sum / (tf.cast(lengths_reshape, dtype=tf.float32) + 1e-30)
    
    return red_avg


# # 定义损失函数和训练函数

# In[4]:

@tf.function
def compute_loss(logits, labels, seqlen):
    """计算序列的交叉熵损失（考虑变长序列）
    
    Args:
        logits: 模型输出，形状(batch_size, seq_len, vocab_size)
        labels: 真实标签，形状(batch_size, seq_len)
        seqlen: 每个序列的实际长度    
        
    Returns:
        平均损失值
    """
    # 计算每个位置的交叉熵损失
    # sparse_softmax_cross_entropy_with_logits 直接接受标签索引，无需one-hot编码
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    
    # 对变长序列求平均（忽略填充部分）
    losses = reduce_avg(losses, seqlen, dim=1)
    
    return tf.reduce_mean(losses)

@tf.function
def train_one_step(model, optimizer, x, y, seqlen):
    """执行一步训练
    
    Args:
        model: 诗歌生成模型
        optimizer: 优化器
        x: 输入序列
        y: 目标序列
        seqlen: 序列长度
        
    Returns:
        当前步骤的损失值
    """
    with tf.GradientTape() as tape:
        # 1. 前向传播
        logits = model(x)
        
        # 2. 计算损失，考虑序列长度掩码
        loss = compute_loss(logits, y, seqlen)
    
    # 3. 反向传播，计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 4. 更新参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

def train(epoch, model, optimizer, ds):
    """训练一个epoch
    
    Args:
        epoch: 当前epoch编号
        model: 诗歌生成模型
        optimizer: 优化器
        ds: 训练数据集
        
    Returns:
        最后一个批次的损失值
    """
    loss = 0.0
    
    # 遍历数据集
    for step, (x, y, seqlen) in enumerate(ds):
        # 训练一步
        loss = train_one_step(model, optimizer, x, y, seqlen)

        # 定期打印损失，监控训练进度
        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy())

    return loss


# # 训练过程

# In[5]:

# 初始化优化器，使用Adam优化算法和较小的学习率
optimizer = optimizers.Adam(0.0005)  # 学习率0.0005

# 加载数据集
train_ds, word2id, id2word = poem_dataset()

# 初始化模型
model = myRNNModel(word2id)

# 训练10个epoch
for epoch in range(10):
    loss = train(epoch, model, optimizer, train_ds)

# # 诗歌生成

# In[74]:

def gen_sentence(model: myRNNModel, word2id: dict, id2word: dict, max_len: int = 50) -> str:
    """使用训练好的RNN模型生成诗歌

    Args:
        model: 训练好的诗歌生成模型
        word2id: 词语到id的映射字典
        id2word: id到词语的映射字典
        max_len: 生成诗歌的最大长度，默认为50

    Returns:
        str: 生成的诗歌字符串（不包含开始和结束标记）
    """
    # 初始化RNN隐藏状态（SimpleRNN只有一个隐藏状态）
    state = tf.random.normal(shape=(1, 128), stddev=0.5)

    # 初始化当前token为开始标记
    cur_token = tf.constant([word2id[start_token]], dtype=tf.int32)
    generated_tokens = []

    # 循环生成token，直到达到最大长度或生成结束标记
    for _ in range(max_len):
        # 获取下一个token并更新RNN状态
        cur_token, state = model.get_next_token(cur_token, state)
        token_id = cur_token.numpy()[0]  # 提取标量ID
        generated_tokens.append(token_id)  # 记录生成的ID

        # 检查是否生成了结束标记，如果是则提前结束
        if id2word[token_id] == end_token:
            break

    # 将生成的token ID序列转换为词语字符串
    # 去除第一个开始标记和最后一个结束标记（如果存在）
    return ''.join([id2word[t] for t in generated_tokens[1:-1]])  # 去除开始和结束标记

# 生成并打印诗歌
print(''.join(gen_sentence(model, word2id, id2word)))
