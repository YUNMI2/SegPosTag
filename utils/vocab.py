import collections
from itertools import chain

import torch
import torch.nn.init as init


class Vocab(object):
    def __init__(self, word_freq, label_freq, min_freq=1):
        words = [word for word, freq in word_freq.items() if freq > min_freq]
        labels = list(label_freq.keys()) 
        
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"
           
        self._words = [self.PAD] + words + [self.UNK]
        self._labels = labels

        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_words = len(self._words)
        self.num_labels = len(self._labels)

        self.UNK_word_index = self._word2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]

    '''
    def collect(self, corpus, min_freq=1):
        labels = sorted(set(chain(*corpus.label_seqs)))
        words = list(chain(*corpus.word_seqs))
        
        # chars_freq = collections.Counter(''.join(words))
        # chars = [c for c,f in chars_freq.items() if f > min_freq]
        words_freq = collections.Counter(words)
        words = [w for w,f in words_freq.items() if f> min_freq]
       
        #return words, chars, labels
        return words, labels

    def __init__(self, train_corpus, min_freq=1):
        # words, chars, labels = self.collect(train_corpus, min_freq)
        words, labels = self.collect(train_corpus, min_freq)
        #  ensure the <PAD> index is 0
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self._words = [self.PAD] + words + [self.UNK]
        # self._chars = [self.PAD] + chars + [self.UNK]
        self._labels = labels

        self._word2id = {w: i for i, w in enumerate(self._words)}
        # self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_words = len(self._words)
        # self.num_chars = len(self._chars)
        self.num_labels = len(self._labels)

        self.UNK_word_index = self._word2id[self.UNK]
        # self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        # self.PAD_char_index = self._char2id[self.PAD]
    '''
   
    '''
    def read_embedding(self, embedding_file, unk_in_pretrain=None):
        #  ensure the <PAD> index is 0
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        if isinstance(unk_in_pretrain, str):
            assert unk_in_pretrain in words
            words = list(words)
            words[words.index(unk_in_pretrain)] = self.UNK

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        out_train_words = [w for w in words if w not in self._word2id]
        out_train_chars = [c for c in ''.join(out_train_words) if c not in self._char2id]

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._words =[self.PAD] + sorted(set(self._words + out_train_words) - {self.PAD})
        self._chars =[self.PAD] + sorted(set(self._chars + out_train_chars) - {self.PAD})

        # update the words,chars dictionary
        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self.UNK_word_index = self._word2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]
        
        # update the numbers of words and chars
        self.num_words = len(self._words)
        self.num_chars = len(self._chars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        extended_embed = torch.randn(self.num_words, embdim)
        init.normal_(extended_embed, 0, 1 / embdim ** 0.5)
        
        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._words):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
        return extended_embed
    '''

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            # get(key ,default) 根据key获取对应的value,如果没有就返回默认值
            return self._word2id.get(str, self.UNK_word_index)
        elif isinstance(word, list):
            return [self._word2id.get(w, self.UNK_word_index) for w in word]

    def label2id(self, label):
        assert (isinstance(label, str) or isinstance(label, list))
        if isinstance(label, str):
            return self._label2id.get(label, 0) # if label not in training data, index to 0 ?
        elif isinstance(label, list):
            return [self._label2id.get(l, 0) for l in label]

    '''
    def char2id(self, char, max_len=20):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.UNK_char_index)
        elif isinstance(char, list):
            return [[self._char2id.get(c, self.UNK_char_index) for c in w[:max_len]] + 
                    [0] * (max_len - len(w)) for w in char]
    '''

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            return self._labels[id]
        elif isinstance(id, list):
            return [self._labels[i] for i in id]
    
    def id2word(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            return self._words[id]
        elif isinstance(id, list):
            return [self._words[i] for i in id]
