import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *

from module import *


class Char_LSTM_CRF(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden, n_word, word_dim,
                 n_layers, word_hidden, n_target, drop=0.5):
        super(Char_LSTM_CRF, self).__init__()

        self.embedding_dim = word_dim
        self.drop1 = torch.nn.Dropout(drop)
        self.embedding = torch.nn.Embedding(n_word, word_dim, padding_idx=0)

        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)

        if n_layers > 1:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=n_layers,
                dropout=0.2
            )
        else:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
        # self.hidden = nn.Linear(word_hidden, word_hidden//2, bias=True)
        self.out = torch.nn.Linear(word_hidden, n_target, bias=True)
        self.crf = CRFlayer(n_target)

        self.reset_parameters()

    def load_pretrained_embedding(self, pre_embeddings):
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.embedding.weight = nn.Parameter(pre_embeddings)

    def reset_parameters(self):
        init.xavier_uniform_(self.out.weight)
        # init.xavier_uniform_(self.hidden.weight)
        # bias = (3.0 / self.embedding.weight.size(1)) ** 0.5
        # init.uniform_(self.embedding.weight, -bias, bias)
        init.normal_(self.embedding.weight, 0, 1 / self.embedding_dim ** 0.5)

    def forward(self, word_idxs, char_idxs):
        # mask = torch.arange(x.size()[1]) < lens.unsqueeze(-1)
        mask = word_idxs.gt(0)
        print("word_idxs:",word_idxs, "\n", word_idxs.size(),"\n")
        print("mask:", mask,"\n", mask.size(), "\n")
        #print(mask[0])
        #print(mask[1])
        #print(mask[2])

        sen_lens = mask.sum(1)
        print("sen_lens:", sen_lens, "\n", sen_lens.size(), "\n")
        
        print("char_idxs:")
        print(char_idxs)
        print()
        print("char_idxs[mask:]")
        print(char_idxs[mask])

        print("mask前后的比较：",torch.equal(char_idxs, char_idxs[mask]),"\n")
        print("char_idxs.size = ", char_idxs.size(),"\t", char_idxs[mask].size(),"\n")
        print()
        char_vec = self.char_lstm.forward(char_idxs[mask])
        #print(char_vec)
        print(char_vec.size())
        print()
        char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True, padding_value=0)
	    #print(char_vec.size())
        exit()

        word_vec = self.embedding(word_idxs)
        feature = self.drop1(torch.cat((word_vec, char_vec), -1))

        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        feature = feature[sorted_idx]
        feature = pack_padded_sequence(feature, sorted_lens, batch_first=True)

        r_out, state = self.lstm_layer(feature, None)
        out, _ = pad_packed_sequence(r_out, batch_first=True, padding_value=0)
        out = out[reverse_idx]
        # out = torch.tanh(self.hidden(out))
        out = self.out(out)
        return out

    def forward_batch(self, batch):
        word_idxs, char_idxs, label_idxs = batch
        mask = word_idxs.gt(0)
        out = self.forward(word_idxs, char_idxs)
        return mask, out, label_idxs

    def get_loss(self, emit, labels, mask):
        emit = emit.transpose(0, 1)
        labels = labels.t()
        mask = mask.t()
        logZ = self.crf.get_logZ(emit, mask)
        scores = self.crf.score(emit, labels, mask)
        return (logZ - scores) / emit.size()[1]
