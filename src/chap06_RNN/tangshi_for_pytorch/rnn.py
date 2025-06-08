import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# 对神经网络中的线性层（Linear）进行权重初始化
def weights_init(m):
    """
    使用 Xavier 均匀分布初始化线性层权重
    
    参数:
        m: 神经网络模块
    """
    classname = m.__class__.__name__  # 获取模块类名
    
    # 仅处理线性层
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())  # 获取权重张量的形状: [输出维度, 输入维度]
        fan_in = weight_shape[1]                   # 输入维度
        fan_out = weight_shape[0]                  # 输出维度
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # 计算 Xavier 初始化的边界
        
        # 使用均匀分布初始化权重
        m.weight.data.uniform_(-w_bound, w_bound)
        
        # 偏置初始化为零
        if m.bias is not None:
            m.bias.data.fill_(0)
            
        print("initialized linear weight")


# 定义词嵌入模块
class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        """
        词嵌入模块，将词索引转换为词向量
        
        参数:
            vocab_length: 词汇表大小
            embedding_dim: 词向量维度
        """
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        
        # PyTorch默认使用均匀分布初始化，范围为[-sqrt(1/dim), sqrt(1/dim)]
        # 如果需要特定范围，可以使用：
        # nn.init.uniform_(self.word_embedding.weight, -1.0, 1.0)

    def forward(self, input_sentence):
        """
        将词索引转换为词向量
        
        参数:
            input_sentence: 词索引张量，形状为 [序列长度] 或 [批次大小, 序列长度]
            
        返回:
            词向量张量，形状为 [序列长度, 嵌入维度] 或 [批次大小, 序列长度, 嵌入维度]
        """
        return self.word_embedding(input_sentence)


# 定义RNN模型
class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        """
        古诗生成模型的核心RNN架构
        
        参数:
            batch_sz: 批次大小
            vocab_len: 词汇表大小
            word_embedding: 预训练的词嵌入模块
            embedding_dim: 词向量维度
            lstm_hidden_dim: LSTM隐藏状态维度
        """
        super(RNN_model, self).__init__()

        # 模型参数设置
        self.word_embedding_lookup = word_embedding  # 使用外部传入的词嵌入模块
        self.batch_size = batch_sz                   # 批次大小
        self.vocab_length = vocab_len                # 词汇表大小
        self.word_embedding_dim = embedding_dim      # 词向量维度
        self.lstm_dim = lstm_hidden_dim              # LSTM隐藏状态维度

        # 定义LSTM层
        # input_size: 输入特征维度（词向量维度）
        # hidden_size: LSTM隐藏状态维度
        # num_layers: LSTM层数（堆叠多个LSTM）
        # batch_first: 如果为True，则输入和输出张量的形状为(batch, seq, feature)
        #              此处为False，表示形状为(seq, batch, feature)
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=False
        )

        # 定义全连接层，将LSTM的输出映射到词汇表大小
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)

        # 应用权重初始化函数
        self.apply(weights_init)

        # 定义log softmax激活函数，用于输出词概率分布
        self.softmax = nn.LogSoftmax(dim=1)  # 在维度1（词汇表维度）上进行softmax

    def forward(self, sentence, is_test=False):
        """
        模型前向传播过程
        
        参数:
            sentence: 输入的词索引序列，形状为 [序列长度]
            is_test: 是否为测试模式
            
        返回:
            预测的词概率分布，形状为 [序列长度, 词汇表大小]（训练模式）
            或 [1, 词汇表大小]（测试模式）
        """
        # 查找词向量
        # 输入形状: [序列长度]
        # 输出形状: [序列长度, 嵌入维度]
        embedded = self.word_embedding_lookup(sentence)
        
        # 调整形状为 (序列长度, 1, 嵌入维度)，适应LSTM的输入要求
        # 其中1表示批次大小为1
        batch_input = embedded.view(-1, 1, self.word_embedding_dim)

        # 初始化LSTM的隐藏状态和细胞状态
        # 形状为 (层数, 批次大小, 隐藏维度)
        h0 = torch.zeros(2, 1, self.lstm_dim)  # 初始隐藏状态
        c0 = torch.zeros(2, 1, self.lstm_dim)  # 初始细胞状态

        # 如果有GPU，将数据移至GPU
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            batch_input = batch_input.cuda()

        # LSTM前向传播
        # output形状: (序列长度, 批次大小, 隐藏维度)
        # hn, cn形状: (层数, 批次大小, 隐藏维度)
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))

        # 重塑输出以适应全连接层
        # 形状变为: (序列长度 * 批次大小, 隐藏维度)
        out = output.contiguous().view(-1, self.lstm_dim)

        # 通过全连接层和ReLU激活函数
        # 形状变为: (序列长度 * 批次大小, 词汇表大小)
        out = F.relu(self.fc(out))

        # 应用log softmax获取词概率分布
        out = self.softmax(out)

        # 测试模式下，仅返回最后一个时间步的预测
        if is_test:
            # 取最后一个时间步的预测
            # 形状: (1, 词汇表大小)
            prediction = out[-1, :].view(1, -1)
            return prediction
        else:
            return out
