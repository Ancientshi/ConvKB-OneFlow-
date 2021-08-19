import math
import numpy as np
from numpy.random import rand
from numpy import zeros
from six import b
import oneflow as flow
import oneflow.nn as nn


class MyConvKB(nn.Module):
    def __init__(self,embeddingDim=200,kernelNum=100):
        '''
        :param embeddingDim: entityEmbedding 与 relationEmbedding的维度
        :param kernelNum: 卷积核的个数
        '''
        super().__init__()
        self.embeddingDim=embeddingDim
        self.conv=nn.Conv1d(in_channels=3,out_channels=kernelNum,kernel_size=1,stride=1,bias=True)
        self.fc=nn.Linear(kernelNum*embeddingDim,1,bias=False)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        conv1d_in=flow.reshape(x,(-1,3,self.embeddingDim))
        #conv1d的输入是(批次大小,单词维度,句子长度)，这里是(batch_size,3,200),conv1d将会在最后一个维度上进行卷积
        #print('conv1d_in.shape',conv1d_in.shape)
        conv1d_out=self.conv(conv1d_in)
        #print('conv1d_out.shape',conv1d_out.shape)
        relu_out=self.relu(conv1d_out)
        #print('relu_out.shape',relu_out.shape)
        flatten_out=self.flatten(relu_out)
        #print('flatten_out.shape',flatten_out.shape)
        fc_out=self.fc(flatten_out)
        #print('fc_out.shape',fc_out.shape)
        return fc_out
