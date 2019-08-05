import argparse

import torch
import torch.utils.data as Data

from config import config
from model.bilstm_crf import BiLSTM_CRF
from utils import *


def process_data(vocab, dataset, max_word_len=30):
    word_idxs, char_idxs, label_idxs = [], [], []

    for wordseq, labelseq in zip(dataset.word_seqs, dataset.label_seqs):
        _word_idxs = vocab.word2id(wordseq)
        _label_idxs = vocab.label2id(labelseq)

        # _char_idxs = vocab.char2id(wordseq, max_word_len)

        word_idxs.append(torch.tensor(_word_idxs)) # 将句中每个词都转成ID的list，然后再转成tensor
        # char_idxs.append(torch.tensor(_char_idxs))
        label_idxs.append(torch.tensor(_label_idxs))

    # return TensorDataSet(word_idxs, char_idxs, label_idxs)
    return TensorDataSet(word_idxs, label_idxs)

if __name__ == '__main__':
    # init config
    model_name = 'bilstm_crf'
    config = config[model_name]
    for name, value in vars(config).items():
        print('%s = %s' %(name, str(value)), flush = True)

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:', flush = True)
    print(args, flush = True)
    print()
    # choose GPU and init seed
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %d' % args.gpu, flush = True)
        print('GPU seed = %d' % torch.cuda.initial_seed(), flush = True)
        print('CPU seed = %d' % torch.initial_seed(), flush = True)
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print('CPU seed = %d' % torch.initial_seed(), flush = True)

    # read training , dev and test file
    print('loading three datasets...', flush = True)
    train = Corpus(config.train_file)
    dev = Corpus(config.dev_file)
    test = Corpus(config.test_file)

    # collect all words, characters and labels in trainning data
    vocab = Vocab(train, min_freq=1)


    # choose if use pretrained word embedding
    # if args.pre_emb and config.embedding_file !=None:
    #     print('loading pretrained embedding...')
    #     pre_embedding = vocab.read_embedding(config.embedding_file)
    # print('Words : %d，Characters : %d，labels : %d' %
    #       (vocab.num_words, vocab.num_chars, vocab.num_labels))
    print('Words : %d, labels : %d'
          %(vocab.num_words, vocab.num_labels), flush = True)
    #print(vocab)
    save_pkl(vocab, config.vocab_file)

    # exit()
    # process training data , change string to index
    print('processing datasets...', flush = True)
    train_data = process_data(vocab, train, max_word_len=20)
    dev_data = process_data(vocab, dev, max_word_len=20)
    test_data = process_data(vocab, test, max_word_len=20)
    print('finish processing datasets...', flush = True)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    #exit()
    # create neural network
    # net = Char_LSTM_CRF(vocab.num_chars,
    #                     config.char_dim,
    #                     config.char_hidden,
    #                     vocab.num_words,
    #                     config.word_dim,
    #                     config.layers,
    #                     config.word_hidden,
    #                     vocab.num_labels,
    #                     config.dropout
    #                     )
    net = BiLSTM_CRF(vocab.num_words,
                     config.word_dim,
                     config.layers,
                     config.word_hidden,
                     vocab.num_labels,
                     config.dropout,
    )
    '''
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding)
    '''
    print(net)

    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        net.cuda()

    # init evaluator
    evaluator = Evaluator(vocab, config)
    # init trainer
    trainer = Trainer(net, config)
    # start to train
    trainer.train((train_loader, dev_loader, test_loader), evaluator)
