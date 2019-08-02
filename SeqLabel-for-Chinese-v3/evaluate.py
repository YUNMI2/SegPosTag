import argparse
import datetime

import torch
import torch.utils.data as Data

from config import config
from model import Char_LSTM_CRF, BiLSTM_CRF
from train import process_data
from utils import *

if __name__ == '__main__':
    # init config
    model_name = 'bilstm_crf'
    #model_name = 'char_lstm_crf'
    config = config[model_name]

    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:')
    print(args)

    # choose GPU
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        print('using GPU device : %d' % args.gpu)
    else:
        torch.set_num_threads(args.thread)
        use_cuda = False

    # loading vocab
    vocab = load_pkl(config.vocab_file)
    # loading network
    print("loading model...")
    network = torch.load(config.net_file)
    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        network.cuda()
    else:
        network.cpu()
    print('loading three datasets...')
    test = Corpus(config.test_file)
    # process test data , change string to index
    print('processing datasets...')
    test_data = process_data(vocab, test, max_word_len=30)
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    # init evaluator
    evaluator = Evaluator(vocab)
    print('evaluating test data...')

    time_start = datetime.datetime.now()
    test_loss, test_p = evaluator.eval(network, test_loader)
    print('test  : loss = %.4f  precision = %.4f' % (test_loss, test_p))
    time_end = datetime.datetime.now()
    print('iter executing time is ' + str(time_end - time_start))
