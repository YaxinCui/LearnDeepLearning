
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
    def __init__(self, tokenizer, max_length, lstm_input_size, lstm_output_size, lstm_num_layers, lstm_bidirection, num_labels, device):
        super(IMDBLstm, self).__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirection = lstm_bidirection
        self.device = device
        self.PAD_TOKEN = tokenizer.vocab2index['<PAD>']
        
        self.embed_shape = tokenizer.vocabVector.shape
        self.embedding = nn.Embedding(self.embed_shape[0], self.embed_shape[1], padding_idx=0)
#        self.batchNorm1 = nn.BatchNorm2d(2,affine=True)
        self.dropout = nn.Dropout(0.2)

        self.LSTM = nn.LSTM(lstm_input_size, lstm_output_size, lstm_num_layers, bidirectional=lstm_bidirection)
        # nn.init.orthogonal_(self.LSTM.weight)
        
        
        self.linear = nn.Linear(lstm_output_size * (int(lstm_bidirection) + 1) * lstm_num_layers, num_labels)
        nn.init.xavier_uniform_(self.linear.weight)
        
        
        # h0 (num_layers * direction, batch, hidden_size)
        # c0 (num_layers * direction, batch, hidden_size)
        
        # output (seq_len, batch, hidden_size * num_directions)
        # hn (num_layers * direction, batch, hidden_size)
        # cn (num_layers * direction, batch, hidden_size )
    
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
        
        packed = pack_padded_sequence(input=embeds, lengths=inputLengths, enforce_sorted=False)
        
        if lstm_h is None or lstm_c is None:
            lstm_outputs, (hn, cn) = self.LSTM(packed, self.initHC(embeds.size(1)))
        else:
            lstm_outputs, (hn, cn) = self.LSTM(packed, (lstm_h, lstm_c))

        # lstm_outputs = pad_packed_sequence(lstm_outputs, padding_value=0.0)
        # lstm_outputs torch.Size([seq_len, batch_size, word_dim * bidirection])
        # hn torch.Size([num_layers * bidirection, batch_size, word_dim])
        # cn torch.Size([num_layers * bidirection, batch_size, word_dim])
        
        outputs = hn.transpose(0, 1).reshape(-1, self.lstm_num_layers * (int(self.lstm_bidirection) + 1) * self.lstm_output_size)
        
        outputs = self.linear(F.relu(outputs))
        
        outputs = F.log_softmax(outputs, dim=1)
        
        return outputs
        
    def initHC(self, batchsize):
        # 默认batch_first = False
        h0=Variable(torch.randn(self.lstm_num_layers * (int(self.lstm_bidirection) + 1), batchsize, self.lstm_output_size)).to(self.device)
        c0=Variable(torch.randn(self.lstm_num_layers * (int(self.lstm_bidirection) + 1), batchsize, self.lstm_output_size)).to(self.device)
        
        return (h0, c0)