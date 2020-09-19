
# IMDB模型
# 为了统一模型接口，模型的输入都是batch 句子，也就是一个句子List，输出为各个句子为不同种类的概率

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import zip_longest

# 设置随机数

class IMDBLstm(nn.Module):
    def __init__(self, tokenizer, max_length, lstm_input_size, lstm_output_size, lstm_bidirection, num_labels, device):
        super(IMDBLstm, self).__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.lstm_bidirection = lstm_bidirection
        self.device = device
        self.PAD_TOKEN = tokenizer.vocab2index['<PAD>']
        
        self.embed_shape = tokenizer.vocabVector.shape
        self.embedding = nn.Embedding(self.embed_shape[0], self.embed_shape[1], padding_idx=0)
        self.dropout = nn.Dropout(0.2)

        self.GRU = nn.GRU(lstm_input_size, lstm_output_size, bidirectional=lstm_bidirection)
        
        self.linear = nn.Linear(lstm_output_size * (int(lstm_bidirection) + 1), num_labels)
    
    def forward(self, inputs, lstm_h=None, lstm_c=None):
        
        tokens = self.tokenizer.tokenizerBatch(inputs)
        inputLengths = [len(token) for token in tokens]
        
        tokens = list(zip_longest(*tokens, fillvalue=self.PAD_TOKEN))
        
        if max(inputLengths) > self.max_length:
            tokens = tokens[: self.max_length]
        
        inputLengths = [length if length < self.max_length else self.max_length for length in inputLengths]
        
        tokens = torch.LongTensor(tokens).to(self.device)
        embeds = self.embedding(tokens)
        # embeds = self.batchNorm1(embeds)
        embeds = self.dropout(embeds)
        
        # (seq_len, batch_size, input_dim)
        packed = pack_padded_sequence(input=embeds, lengths=inputLengths, enforce_sorted=False)
        
        # (seq_len, batch_size, output_dim)
        lstm_outputs = self.GRU(packed)[0]
        outputs = pad_packed_sequence(lstm_outputs, total_length=embeds.size(0))[0]
        
        # Maxpooling
        # (seq_len, batch_size, output_dim) -> (batch_size, seq_len, output_dim) -> (batch_size, output_dim)
        outputs = torch.max(outputs.transpose(0, 1), dim=1)[0]
        # (batch_size, output_dim) -> (batch_size, class_num)
        outputs = self.linear(outputs)
        
        outputs = F.log_softmax(outputs, dim=1)
        
        return outputs
        
    def initHC(self, batchsize):
        # 默认batch_first = False
        h0=Variable(torch.randn(self.lstm_num_layers * (int(self.lstm_bidirection) + 1), batchsize, self.lstm_output_size)).to(self.device)
        c0=Variable(torch.randn(self.lstm_num_layers * (int(self.lstm_bidirection) + 1), batchsize, self.lstm_output_size)).to(self.device)
        
        return (h0, c0)