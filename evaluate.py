import argparse
import datetime

import torch
import torch.utils.data as Data

from config import config
from model.bilstm_crf import BiLSTM_CRF
from train import process_data
from utils import *
from utils.mylib import *

if __name__ == '__main__':
    # init config
    model_name = 'bilstm_crf'
    #model_name = 'char_lstm_crf'
    config = config[model_name]

    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:', flush =True)
    print(args, flush =True)

    # choose GPU
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        print('using GPU device : %d' % args.gpu, flush =True)
    else:
        torch.set_num_threads(args.thread)
        use_cuda = False

    # loading vocab
    vocab = load_pkl(config.vocab_file)
    # loading network
    print("loading model...", flush =True)
    network = torch.load(config.net_file)
    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        network.cuda()
    else:
        network.cpu()
    print('loading three datasets...', flush =True)
    test = Corpus(config.eval_file)
    # process test data , change string to index
    print('processing datasets...', flush =True)
    test_data = process_data(vocab, test, max_word_len=30)
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    # init evaluatior
    evaluator = Evaluator(vocab, config)
    print('evaluating test data...', flush =True)

    time_start = datetime.datetime.now()
    with torch.no_grad():
        test_loss, test_p, test_r, test_f, test_word, test_predict, test_target = evaluator.eval(network, test_loader)
        print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss, test_p, test_r, test_f), flush = True)
    if config.predictOut:
        writeConll(config.predict_eval_file, test_word, test_predict, test_target)
    #print('test  : loss = %.4f  precision = %.4f' % (test_loss, test_p))
    time_end = datetime.datetime.now()
    print('iter executing time is ' + str(time_end - time_start), flush =True)
