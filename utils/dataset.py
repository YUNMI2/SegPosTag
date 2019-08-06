import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)

def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


def collate_fn_cuda(data):
    batch = zip(*data)
    return tuple([torch.tensor(x).cuda() if len(x[0].size()) < 1 else pad_sequence(x, True).cuda() for x in batch])



def process_data(vocab, word_seqs, label_seqs, max_word_len=30):
    word_idxs, label_idxs = [], []

    for wordseq, labelseq in zip(word_seqs, label_seqs):
        _word_idxs = vocab.word2id(wordseq)
        _label_idxs = vocab.label2id(labelseq)

        word_idxs.append(torch.tensor(_word_idxs)) # 将句中每个词都转成ID的list，然后再转成tensor
        label_idxs.append(torch.tensor(_label_idxs))

    return TensorDataSet(word_idxs, label_idxs)
