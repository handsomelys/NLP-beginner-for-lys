from multiprocessing import pool
from turtle import forward
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter, filter_sizes,
    output_dim, dropout = 0.2, pad_idx = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        # in_channels: 输入的 channel 文字都是1
        # out_channels: 输出的 channel 维度
        # fs: 每次滑动窗口计算用到几个单词 相当于 n-gram 中的 n 
        # for fs in filter_sizes 用好几个卷积模型 最后 concate 起来看效果
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=num_filter,
            kernel_size=(fs,embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes)*num_filter,output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        embedded = self.dropout(self.embedding(text)) # [batch size, sentence len, emb dim]
        embedded = embedded.unsqueeze(1)   # [batch size, 1, sentence len, emb dim]
        # 升维 为了和 nn.conv2d 的输入维度吻合 把 channel 列升维
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch sizez, num_filter, sentence len - fileter_sizes + 1]
        # 有几个 fileter_sizes 就有几个 conved

        pooled  = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved] # [batch,num_filter]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filter * len(filter_sizes)]
        # 把 len(filter_sizes) 个卷积模型concate起来传到全连接层

        return self.fc(cat)
