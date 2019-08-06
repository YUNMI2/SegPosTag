import argparse

import torch
import torch.utils.data as Data

from config import config
from model.bilstm_crf import BiLSTM_CRF
from utils import *
from utils.dataset import *
from utils.mylib import *

if __name__ == '__main__':
    # init config
    model_name = 'bilstm_crf'
    config = config[model_name]
    
    
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
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %d' % args.gpu, flush = True)
        print('GPU seed = %d' % torch.cuda.initial_seed(), flush = True)
        print('CPU seed = %d' % torch.initial_seed(), flush = True)
    else:
        config.use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print('CPU seed = %d' % torch.initial_seed(), flush = True)

    for name, value in vars(config).items():
        print('%s = %s' %(name, str(value)), flush = True)
    
    # read training , dev and test file
    print('loading three datasets...', flush = True)
    '''
     if config.train_file.strip():
        train = Corpus(config.train_file)
  
    dev = Corpus(config.dev_file)
    test = Corpus(config.test_file)
    '''
    
    # train_file 要么就是一个，要么就是一个文件夹，为了防止训练数据过大占用过多内存
    assert (config.train_file.strip() and not config.train_files_hold.strip()) or (not config.train_file.strip() and config.train_files_hold.strip())
    train_files = [config.train_file.strip()] if config.train_file.strip() else loadAllFile(config.train_files_hold.strip())  
    dev_files = config.dev_file
    test_files = config.test_file
    
    train_word_freq, train_label_freq = Corpus().stat(train_files , "Train", True)
    dev_word_freq, dev_label_freq = Corpus().stat(dev_files, "Dev", False)
    test_word_freq, test_label_freq = Corpus().stat(test_files, "Test", False)
    
    # collect all words, characters and labels in trainning data
    vocab = Vocab(train_word_freq, train_label_freq, min_freq=1)

    print('Words : %d, labels : %d'
          %(vocab.num_words, vocab.num_labels), flush = True)
    save_pkl(vocab, config.vocab_file)

    # change dev_files to index 
    dev_sentences, dev_labels = Corpus().getWordLabelSeq(dev_files)
    dev_data = process_data(vocab, dev_sentences, dev_labels, max_word_len=20)
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
    )
    
    # change test_files to index 
    test_sentences, test_labels = Corpus().getWordLabelSeq(test_files)
    test_data = process_data(vocab, test_sentences, test_labels, max_word_len=20)
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
    )
    '''
    # 需要重新写
    print('processing datasets...', flush = True)
    train_data = process_data(vocab, train, max_word_len=20)
    dev_data = process_data(vocab, dev, max_word_len=20)
    test_data = process_data(vocab, test, max_word_len=20)
    print('finish processing datasets...', flush = True)
    '''
    '''
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
    '''
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
    if config.use_cuda:
        net.cuda()

    # init evaluator
    evaluator = Evaluator(vocab, config)
    # init trainer
    trainer = Trainer(net, config)
    # start to train
    trainer.train(train_files, (dev_loader, test_loader), evaluator)
