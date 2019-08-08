import argparse
import datetime

import torch
import torch.utils.data as Data

from config import config
from model.bilstm_crf import BiLSTM_CRF
from utils.dataset import process_data
from utils import *
from utils.mylib import *
from train import *

if __name__ == '__main__':
    # init config
    model_name = 'bilstm_crf'
    #model_name = 'char_lstm_crf'
    config = config[model_name]

    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--gpu', type=str, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:', flush =True)
    print(args, flush =True)

    # choose GPU
    if args.gpu:
        config.use_cuda = True
        gpus = [int(x) for x in args.gpu.split(',')]
        assert gpus.__len__() == 1
        torch.cuda.set_device(gpus[0])
        torch.set_num_threads(args.thread)
        print('using GPU device : %d' % gpus[0], flush =True)
    else:
        torch.set_num_threads(args.thread)
        config.use_cuda = False

    # loading vocab
    vocab = load_pkl(config.vocab_file)
    # loading network
    print("loading model...", flush =True)
    net = torch.load(config.net_file)
    # if use GPU , move all needed tensors to CUDA
    if config.use_cuda:
        net.cuda()
    else:
        net.cpu()
    print('loading three datasets...', flush =True)
    #test = Corpus(config.eval_file)
    test_sentences, test_labels = Corpus().getWordLabelSeq(config.eval_file)
    #test_data = process_data(vocab, test_sentences, test_labels, max_word_len=20)
    # process test data , change string to index
    print('processing datasets...', flush =True)
    test_loader = Data.DataLoader(
        dataset=process_data(vocab, test_sentences, test_labels, max_word_len=20),
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
    )
    del test_sentences, test_labels

    # init evaluatior
    #evaluator = Evaluator(vocab, config)
    print('evaluating test data...', flush =True)

    time_start = datetime.datetime.now()
    with torch.no_grad():
        net.eval()
        test_loss = 0.0
        test_evaler = Evaluator("Test", vocab, config, net)
        for batch in test_loader:
             mask, out, targets = forward_batch(net, batch)
             test_loss += net.get_loss(out, targets, mask)
             '''
             if gpus.__len__() > 1:
                 test_loss += net.module.get_loss(out, targets, mask)
             else:
                 test_loss += net.get_loss(out, targets, mask)
             '''
             test_evaler.parse(batch, mask, out, targets)
        test_p, test_r, test_f = test_evaler.eval()
        print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss, test_p, test_r, test_f), flush = True) 
    
    if config.predictOut:
        test_evaler.write(config.predict_eval_file)
    time_end = datetime.datetime.now()
    print('iter executing time is ' + str(time_end - time_start), flush =True)
