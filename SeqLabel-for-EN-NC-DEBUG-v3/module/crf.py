import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class CRFlayer(torch.nn.Module):
    def __init__(self, labels_num):
        super(CRFlayer, self).__init__()
        self.labels_num = labels_num
        # (i,j)=score(tag[i]->tag[j])
        self.transitions = torch.nn.Parameter(torch.randn(labels_num, labels_num))
        # (i)=score(<BOS>->tag[i])
        self.strans = torch.nn.Parameter(torch.randn(labels_num))
        # (i)=score(tag[i]-><EOS>)
        self.etrans = torch.nn.Parameter(torch.randn(labels_num))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.transitions.data, 0,1 / self.labels_num ** 0.5)
        init.normal_(self.strans.data, 0, 1 / self.labels_num ** 0.5)
        init.normal_(self.etrans.data, 0, 1 / self.labels_num ** 0.5)

    def get_logZ(self, emit, mask):
        '''
        emit: emission (unigram) scores of sentences in batch,[sen_len, batch_size, labels_num]
        mask: masks of sentences in batch,[sen_lens, batch_size]
        return: sum(logZ) in batch
        '''
        sen_len, batch_size, labels_num = emit.shape
        assert (labels_num==self.labels_num)

        alpha = emit[0] + self.strans  # [batch_size, labels_num]
        for i in range(1, sen_len):
            trans_i = self.transitions.unsqueeze(0)  # [1, labels_num, labels_num]
            emit_i = emit[i].unsqueeze(1)  # [batch_size, 1, labels_num]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [batch_size, labels_num, labels_num]
            scores = torch.logsumexp(scores, dim=1)  # [batch_size, labels_num]

            mask_i = mask[i].unsqueeze(1).expand_as(alpha)  # [batch_size, labels_num]
            alpha[mask_i] = scores[mask_i]

        return torch.logsumexp(alpha+self.etrans, dim=1).sum()

    def score(self, emit, target, mask):
        '''
        author: zhangyu
        return: sum(score)
        '''
        sen_len, batch_size, labels_num = emit.shape
        assert (labels_num==self.labels_num)

        scores = torch.zeros_like(target, dtype=torch.float)  #[sen_len, batch_size, labels_num]

        # 加上句间迁移分数
        scores[1:] += self.transitions[target[:-1], target[1:]]
        # 加上发射分数
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # 通过掩码过滤分数
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.strans[target[0]].sum()
        # 加上句尾迁移分数
        score += self.etrans[target.gather(dim=0, index=ends)].sum()
        return score

    def forward(self, emit, labels, mask):
        '''
        emit: emission (unigram) scores of sentences in batch, [batch_size, sen_len, labels_num]
        mask: masks of sentences in batch, [batch_size, sen_lens]
        labels: target of sentences, [batch_size, sen_lens]
        return: sum(logZ-score)/batch_size
        '''
        emit = emit.transpose(0, 1)
        labels = labels.t()
        mask = mask.t()
        logZ = self.get_logZ(emit, mask)#这边就是全部
        scores = self.score(emit, labels, mask)
        # return logZ - scores
        return (logZ - scores) / emit.size()[1]
