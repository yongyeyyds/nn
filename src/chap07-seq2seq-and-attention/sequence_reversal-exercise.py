# !/usr/bin/env python
# coding: utf-8

# # 序列逆置
# 使用sequence to sequence 模型将一个字符串序列逆置。
# 例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`(下图来自网络，是一个sequence to sequence 模型示意图 )
# ![seq2seq](./seq2seq.png)

# In[1]:

# 标准库（Python内置模块，按字母顺序排列）
import collections
import os
import sys
import tqdm  # 虽然tqdm是第三方库，但常作为工具库放在标准库后

# 第三方库（按字母顺序排列，优先导入独立库，再导入子模块）
import numpy as np  # 导入NumPy库（科学计算基础库）
                    # 提供多维数组操作、数学函数、线性代数等功能
                    # 常用于数据预处理、模型输入构建和结果分析
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
# 同一库的子模块合并导入，按字母顺序排列
import random
import string

# ## 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

# In[2]:


def random_string(length):
    """
    生成一个由大写英文字母组成的随机字符串。
    
    参数:
        length (int): 要生成的字符串长度。
        
    返回:
        str: 随机生成的字符串。
    """
    # 步骤 1：定义可用字符集（这里使用大写英文字母）
    # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letters = string.ascii_uppercase

    # 步骤 2：从字符集中随机选择指定数量的字符
    # 使用 random.choice(letters) 从 letters 中随机选择一个字符
    random_chars = [random.choice(letters) for _ in range(length)]

    # 步骤 3：将字符列表拼接成字符串并返回
    return ''.join(random_chars)
    # 重复这个过程 stringLength 次，并用 ''.join() 将这些字符连接成一个字符串
    # 最终返回生成的随机字符串

def get_batch(batch_size, length):
    """
    生成一批用于训练的序列数据
    
    参数:
        batch_size (int): 批次大小
        length (int): 序列长度
        
    返回:
        tuple: 包含四个元素的元组
            - batched_examples: 原始字符串列表
            - enc_x: 编码器输入的索引序列 [batch_size, length]
            - dec_x: 解码器输入的索引序列 [batch_size, length]
            - y: 解码器期望输出的索引序列 [batch_size, length]
    """
    # 生成batch_size个随机字符串
    batched_examples = [random_string(length) for i in range(batch_size)]
    
    # 转成索引：A->1, B->2, ..., Z->26
    enc_x = [[ord(ch) - ord('A') + 1 for ch in list(exp)] for exp in batched_examples]
    
    # 逆序：目标输出是输入的逆序
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    
    # 添加起始符：解码器输入以特殊起始符(0)开始，后面接目标序列的前n-1个元素
    dec_x = [[0] + e_idx[:-1] for e_idx in y]
    
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))

print(get_batch(2, 10))

###

# # 建立sequence to sequence 模型##

# In[3]:


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        # 初始化父类 keras.Model，必须调用
        super().__init__()

        # 词表大小为27：A-Z共26个大写字母，加上1个特殊的起始符（用0表示）
        self.v_sz = 27 # 词表大小：26个字母+1个起始符（0）

        # 嵌入层：将每个字符的索引映射成64维的向量表示
        # 输入维度：self.v_sz（即词表大小），输出维度为64
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64,
                                                    batch_input_shape=[None, None])

        # 编码器RNN单元：使用SimpleRNNCell，隐藏状态维度为128
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(128)

        # 解码器RNN单元：使用SimpleRNNCell，隐藏状态维度为128
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(128)

        # 编码器RNN层：将RNNCell包裹成完整RNN
        # return_sequences=True：返回整个序列的输出
        # return_state=True：返回最终状态，用于初始化解码器
        self.encoder = tf.keras.layers.RNN(
            self.encoder_cell,
            return_sequences=True,  # 返回每个时间步的输出
            return_state=True       # 返回最终隐藏状态
        )

        # 解码器RNN层：与编码器类似
        self.decoder = tf.keras.layers.RNN(
            self.decoder_cell,             # 指定解码器使用的RNN单元
            return_sequences=True,         # 返回完整的输出序列
            return_state=True              # 返回最终的隐藏状态，用于后续时间步
        )

        # 全连接层：将解码器的输出转换为词表大小的logits
        # 用于预测每个位置的字符概率分布
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
        # 注意力机制所需的全连接层
        self.dense_attn = tf.keras.layers.Dense(128)
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        完成sequence2sequence 模型的搭建，模块已经在`__init__`函数中定义好
        前向传播过程：编码器 -> 解码器 -> 全连接层
        
        参数:
            enc_ids: 编码器输入序列 [batch_size, enc_seq_len]
            dec_ids: 解码器输入序列 [batch_size, dec_seq_len]
            
        返回:
            logits: 预测结果 [batch_size, dec_seq_len, vocab_size]
        '''
        # 编码过程
        enc_emb = self.embed_layer(enc_ids)  # (batch_size, enc_seq_len, emb_dim)
        enc_out, enc_state = self.encoder(enc_emb)  # enc_out: (batch_size, enc_seq_len, enc_units)
        
        # 解码过程，使用编码器的最终状态作为初始状态
        dec_emb = self.embed_layer(dec_ids)  # (batch_size, dec_seq_len, emb_dim)
        dec_out, dec_state = self.decoder(dec_emb, initial_state=enc_state)  # dec_out: (batch_size, dec_seq_len, dec_units)
        
        # 计算logits，转换为词表大小的输出
        logits = self.dense(dec_out)  # (batch_size, dec_seq_len, vocab_size)
        
        return logits
    
    
    @tf.function
    def encode(self, enc_ids):
        """
        对输入序列进行编码，获取编码器的最终状态
        
        参数:
            enc_ids: 输入序列 [batch_size, seq_len]
            
        返回:
            list: 包含编码器输出和最终状态的列表
        """
        # 通过嵌入层将token ID转换为词向量
        enc_emb = self.embed_layer(enc_ids)  # shape(b_sz, len, emb_sz)
        
        # 使用编码器处理嵌入向量，获取编码器输出和最终状态
        enc_out, enc_state = self.encoder(enc_emb)
        
        # 返回编码器最后一个时间步的输出和最终状态
        return [enc_out[:, -1, :], enc_state]
    
    def get_next_token(self, x, state, enc_out=None):
        '''
        根据当前输入和状态生成下一个token，带注意力机制
        
        参数:
            x: 当前输入token，shape=[b_sz,]
            state: 当前RNN状态
            enc_out: 编码器输出，用于注意力计算
            
        返回:
            tuple: 包含预测的下一个token和更新后的RNN状态
        '''
        # 将输入token通过嵌入层转换为密集向量表示
        x_embed = self.embed_layer(x)  # (B, E)
        
        if enc_out is not None:
            # 加性注意力计算
            # 计算注意力分数
            score = tf.nn.tanh(self.dense_attn(enc_out))  # (B, T1, H)
            
            # 计算注意力权重
            # 将注意力得分转换为概率分布：通过softmax函数确保权重和为1
            score = tf.reduce_sum(score * tf.expand_dims(state, 1), axis=-1)  # (B, T1)
            attn_weights = tf.nn.softmax(score, axis=-1)  # (B, T1)
            
            # 计算上下文向量：对编码器输出按注意力权重加权求和
            context = tf.reduce_sum(enc_out * tf.expand_dims(attn_weights, -1), axis=1)  # (B, H)
            
            # 将嵌入向量和上下文向量拼接作为RNN输入
            rnn_input = tf.concat([x_embed, context], axis=-1)  # (B, E+H)
        else:
            # 无注意力机制时，直接使用嵌入向量作为RNN输入
            rnn_input = x_embed
            
        # 通过RNN单元计算输出和更新状态
        output, new_state = self.decoder_cell(rnn_input, [state])  # SimpleRNNCell返回单个状态
        
        # 通过全连接层计算logits
        logits = self.dense(output)  # (B, V)
        
        # 选择概率最高的token作为预测结果
        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B,)
        
        return next_token, new_state[0]  # 返回单个状态向量


# # Loss函数以及训练逻辑

# In[4]:

# 定义了一个使用TensorFlow的@tf.function装饰器的函数compute_loss，用于计算模型预测的损失值
@tf.function
def compute_loss(logits, labels):
    """计算交叉熵损失"""
    # 计算稀疏交叉熵损失
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    
    # 计算平均损失
    losses = tf.reduce_mean(losses)
    
    return losses

# 定义了一个使用TensorFlow的@tf.function装饰器的函数train_one_step，用于执行一个训练步骤
@tf.function  # 将函数编译为TensorFlow计算图，提升性能
def train_one_step(model, optimizer, enc_x, dec_x, y):
    """执行一次训练步骤（前向传播+反向传播）"""
    # 自动记录梯度
    with tf.GradientTape() as tape:
        # 前向传播获取预测值
        logits = model(enc_x, dec_x)
        
        # 计算预测值与标签的损失
        loss = compute_loss(logits, y)

    # 计算梯度（自动微分）
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 应用梯度更新模型参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 返回当前步骤的损失值
    return loss

def train(model, optimizer, seqlen):
    """训练过程，迭代 3000 步"""
    # 初始化训练指标
    loss = 0.0  # 记录loss值 (初始为0)
    
    for step in range(3000):
        # 获取训练batch数据:
        # - batched_examples: 原始样本 (用于调试/可视化)
        # - enc_x: 编码器输入序列 [batch_size, seqlen]
        # - dec_x: 解码器输入序列 [batch_size, seqlen] 
        # - y: 目标输出序列 [batch_size, seqlen]
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        
        # 执行单步训练并返回当前loss
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        
        # 每500步计算并打印训练进度和准确率
        if step % 500 == 0:
            # 计算训练准确率
            logits = model(enc_x, dec_x)
            preds = tf.argmax(logits, axis=-1)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
            
            print(f'step {step}: loss={loss.numpy():.4f}, acc={acc.numpy():.4f}')
    
    return loss


# # 训练迭代

# In[5]:
optimizer = optimizers.Adam(0.0005)  # 创建一个 Adam 优化器，用于更新模型参数。
model = mySeq2SeqModel()  # 实例化一个序列到序列（Seq2Seq）模型。
train(model, optimizer, seqlen=20)  # 调用 train 函数开启模型训练流程。


# # 测试模型逆置能力
# 首先要先对输入的一个字符串进行encode，然后在用decoder解码出逆置的字符串
# 测试阶段跟训练阶段的区别在于，在训练的时候decoder的输入是给定的，而在预测的时候我们需要一步步生成下一步的decoder的输入

# In[6]:


def sequence_reversal():
    """测试阶段：对一个字符串执行encode，然后逐步decode得到逆序结果"""
    def decode(init_state, steps=10, enc_out=None):
        """
        逐步解码生成序列
        
        参数:
            init_state: 初始状态
            steps: 生成步数
            enc_out: 编码器输出，用于注意力机制
            
        返回:
            list: 生成的字符串列表
        """
        # 获取批次大小
        b_sz = tf.shape(init_state[0])[0]
        
        # 起始 token（全为 0，表示序列开始）
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        
        # 初始化状态为编码器输出的状态
        state = init_state
        
        # 存储每一步生成的token
        collect = []
        
        # 逐步解码生成序列
        for i in range(steps):
            # 获取下一个token预测和更新后的状态
            # 传入编码器输出用于注意力计算
            cur_token, state = model.get_next_token(cur_token, state, enc_out)
            
            # 收集每一步生成的 token
            collect.append(tf.expand_dims(cur_token, axis=-1))
        
        # 拼接输出序列
        out = tf.concat(collect, axis=-1).numpy()
        
        # 将数值索引转换为对应的字母字符串
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out] 
        
        return out
    

    # 生成一批测试数据（32个样本，每个序列长度10）
    batched_examples, enc_x, _, _ = get_batch(32, 10)
    
    # 对输入序列进行编码，获取编码器输出和状态
    enc_output, enc_state = model.encode(enc_x)
    
    # 解码生成逆序序列，步数等于输入序列长度
    # 传入编码器输出用于注意力机制
    return decode(enc_state, enc_x.get_shape()[-1], enc_output), batched_examples

def is_reverse(seq, rev_seq):
    """检查 rev_seq 是否为 seq 的逆序"""
    # 反转rev_seq并与原始seq比较
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    
    if seq == rev_seq_rev:
        return True  # 返回 True 表示预测结果与真实逆序相符
    else:
        return False

# 测试模型逆序能力的准确性
print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
# 列表推导式对 sequence_reversal() 生成的序列对中的每个元素应用 is_reverse() 函数，zip(*sequence_reversal()) 会将两个序列的对应位置元素配对

# 打印 sequence_reversal() 生成的序列对（经过 zip 转置后的结果），这里会显示实际被 is_reverse 函数比较的各个元素对
print(list(zip(*sequence_reversal())))
