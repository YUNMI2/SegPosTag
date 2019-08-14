import argparse
import datetime

import torch
import torch.utils.data as Data

from config import *
from model.bilstm_crf import BiLSTM_CRF
from utils.dataset import process_data
from utils import *
from utils.mylib import *
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--config', type=str, help='file save model and config')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--thread', type=int, default=4, help='thread num')
    parser.add_argument("--eval_file", type=str, help='file to eval')
    parser.add_argument("--predict_file", type=str, help='path to save eval result')
    args = parser.parse_args()
    print('setting:', flush =True)
    print(args, flush =True)

    # loading config
    print("loading conf...", flush =True)
    config = load_pkl(args.config)
    config['useMultiGPU'] = False
    config['useDistGPU'] = False
    for k,v in config.items():
        print("%s = %s "%(k,v), flush=True)


    # loading vocab
    print("loading vocab...", flush =True)
    vocab = load_pkl(config['vocab_file'])

    # loading network
    print("loading model...", flush =True)
    net = torch.load(config['net_file'])

    # choose GPU
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        print('using GPU device : %d' % args.gpu, flush =True)
        config['use_cuda'] = True
        net.cuda()
    else:
        torch.set_num_threads(args.thread)
        config['use_cuda'] = False
        net.cpu()
   
    print('loading test datasets...', flush =True)
    test_sentences, test_labels = Corpus().getWordLabelSeq(args.eval_file)

    print('processing datasets...', flush =True)
    test_loader = Data.DataLoader(
        dataset=process_data(vocab, test_sentences, test_labels, max_word_len=20),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn if not config['use_cuda'] else collate_fn_cuda
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
             test_evaler.parse(batch, mask, out, targets)
        test_p, test_r, test_f = test_evaler.eval()
        print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss, test_p, test_r, test_f), flush = True) 
    
        test_evaler.write(args.predict_file)
    time_end = datetime.datetime.now()
    print('iter executing time is ' + str(time_end - time_start), flush =True)
