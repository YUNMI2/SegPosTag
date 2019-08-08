import argparse
import torch
import torch.utils.data as Data
from config import config
#from model.bilstm_crf import BiLSTM_CRF
from model.bilstm import BiLSTM
from utils import *
from utils.dataset import *
from utils.mylib import *
import os 
import torch.optim as optim
import datetime
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


def forward_batch(net, batch):
    word_idxs, label_idxs = batch
    mask = word_idxs.gt(0)
    max_length = batch[0].size()[1]
    out = net.forward(word_idxs, max_length)
    return mask, out, label_idxs

if __name__ == '__main__':
    # init config
    model_name = 'bilstm'
    config = config[model_name]
    
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', type=str, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:', flush = True)
    print(args, flush = True)
    print()
    # choose GPU and init seed
    if args.gpu:
        config.use_cuda = True
        gpus = [int(x) for x in args.gpu.split(",")]
        if gpus.__len__() > 1:
            config.multiGPU = True 
        torch.cuda.set_device(gpus[0])
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %s' % args.gpu, flush = True)
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
    
    # check fileName
    assert (config.train_file.strip() and not config.train_files_hold.strip()) or (not config.train_file.strip() and config.train_files_hold.strip())
    assert config.dev_file and config.test_file
    
    # get all filenames 
    train_files = [config.train_file.strip()] if config.train_file.strip() else loadAllFile(config.train_files_hold.strip())  
    dev_files = config.dev_file
    test_files = config.test_file
    
    # get stat info, not load all content into memory 
    train_word_freq, train_label_freq = Corpus().stat(train_files , "Train", True)
    dev_word_freq, dev_label_freq = Corpus().stat(dev_files, "Dev", False)
    test_word_freq, test_label_freq = Corpus().stat(test_files, "Test", False)
    
    # build, show and save vocab
    vocab = Vocab(train_word_freq, train_label_freq, min_freq=1)
    print('Words : %d, labels : %d'%(vocab.num_words, vocab.num_labels), flush = True)
    save_pkl(vocab, config.vocab_file)
    
    # clear Memory 
    del train_word_freq, train_label_freq
    del dev_word_freq, dev_label_freq
    del test_word_freq, test_label_freq

    # load and pack dev data  
    dev_sentences, dev_labels = Corpus().getWordLabelSeq(dev_files)
    dev_loader = Data.DataLoader(
        dataset=process_data(vocab, dev_sentences, dev_labels, max_word_len=20),
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
    )
    del dev_sentences, dev_labels
    
    # load and pack test data  
    test_sentences, test_labels = Corpus().getWordLabelSeq(test_files)
    test_loader = Data.DataLoader(
        dataset=process_data(vocab, test_sentences, test_labels, max_word_len=20),
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
    )
    del test_sentences, test_labels

    # build natwork
    if config.use_crf:  
        net = BiLSTM_CRF(vocab.num_words,
                     config.word_dim,
                     config.layers,
                     config.word_hidden,
                     vocab.num_labels,
                     config.dropout,
        )  
    else:
        net = BiLSTM(vocab.num_words,
                     config.word_dim,
                     config.layers,
                     config.word_hidden,
                     vocab.num_labels,
                     config.dropout,
        )
    print(net)

    # init 
    if config.optimizer == 'adam':
         print('Using Adam optimizer...', flush = True)
         optimizer = optim.Adam(net.parameters(), lr=config.lr)
    
    # if use GPU , move all needed tensors to CUDA
    if config.use_cuda:
        # use multi-GPU 
        if gpus.__len__() > 1:
            net = torch.nn.DataParallel(net, device_ids=gpus)
        net.cuda()

    print("Start to Train a model........", flush=True)

    # init some para in train 
    print("init ...", flush=True)
    max_dev_f, final_test_f = 0.0, 0.0
    max_epoch, patience = 0, 0


    for e in range(config.epoch):
        print('--------------------------------------------Epoch<%d>-------------------------------------------- '%(e+1), flush=True)
        net.train()
        time_start = datetime.datetime.now()
        file_num = 0
        for train_file in train_files:
             file_num += 1
             print("%s/%s"%(file_num, train_files.__len__()), end="\t", flush=True)
             train_sentences, train_labels = Corpus().getWordLabelSeq(train_file)
             train_loader = Data.DataLoader(
                  dataset=process_data(vocab, train_sentences, train_labels, max_word_len=20),
                  batch_size=config.batch_size,
                  shuffle=config.shuffle,
                  collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
             )
             del train_sentences, train_labels
             
             for batch in train_loader:
                 optimizer.zero_grad()
                 mask, out, targets = forward_batch(net, batch)
                 
                 if gpus.__len__() > 1:
                     loss = net.module.get_loss(out, targets, mask)
                 else:
                     loss = net.get_loss(out, targets, mask)
                 
                 #backword 
                 loss.backward()
                 nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                 optimizer.step()
       
        print("\n", fulsh=True)
        with torch.no_grad():
            net.eval()
            train_loss, dev_loss, test_loss = 0.0, 0.0, 0.0
            
            train_sentences, train_labels = Corpus().getWordLabelSeq(train_files[0])
            train_loader = Data.DataLoader(
                  dataset=process_data(vocab, train_sentences, train_labels, max_word_len=20),
                  batch_size=config.batch_size,
                  shuffle=False,  
                  collate_fn=collate_fn if not config.use_cuda else collate_fn_cuda
            )                   
            del train_sentences, train_labels

            print("Computing Train Loss.........", flush=True)
            train_evaler = Evaluator("Train", vocab, config, net)
            for batch in train_loader:
                mask, out, targets = forward_batch(net, batch)
                if gpus.__len__() > 1:
                    train_loss += net.module.get_loss(out, targets, mask)  
                else:
                    train_loss += net.get_loss(out, targets, mask)  
                train_evaler.parse(batch, mask, out, targets)
            train_p, train_r, train_f = train_evaler.eval()
            print('Train   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (train_loss, train_p, train_r, train_f), flush = True)
             
            print("Computing Dev Loss..........", flush=True)
            dev_evaler = Evaluator("Dev", vocab, config, net)
            for batch in dev_loader:
                mask, out, targets = forward_batch(net, batch)
                if gpus.__len__() > 1:
                    dev_loss += net.module.get_loss(out, targets, mask)  
                else:
                    dev_loss += net.get_loss(out, targets, mask)  
                
                dev_evaler.parse(batch, mask, out, targets)
            dev_p, dev_r, dev_f = dev_evaler.eval()
            print('dev   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (dev_loss, dev_p, dev_r, dev_f), flush = True)
             
            print("Computing Test Loss..........", flush=True)
            test_evaler = Evaluator("Test", vocab, config, net)
            for batch in test_loader:
                mask, out, targets = forward_batch(net, batch)
                if gpus.__len__() > 1:
                    test_loss += net.module.get_loss(out, targets, mask)  
                else:    
                    test_loss += net.get_loss(out, targets, mask)  
                test_evaler.parse(batch, mask, out, targets)
            test_p, test_r, test_f = test_evaler.eval()
            print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss, test_p, test_r, test_f), flush = True)

        # save the model when dev precision get better
        if dev_f > max_dev_f:
            max_dev_f = dev_f
            max_test_f = test_f
            max_epoch = e + 1
            patience = 0
            print('Ex best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' %(max_epoch, max_dev_f, max_test_f), flush = True)
            print('save the model...', flush = True)
            if config.multiGPU: 
                torch.save(net.module, config.net_file)
            else:
                torch.save(net, config.net_file)
            #torch.save(net, config.net_file)
            if config.predictOut:
                #train_evaler.write("")
                #dev_evaler.write("")
                test_evaler.write(config.predict_test_file)
        else: 
            patience += 1
        del train_evaler, dev_evaler, test_evaler
        
        time_end = datetime.datetime.now()
        print('iter executing time is ' + str(time_end - time_start) + '\n', flush = True)
        if patience > config.patience:
            break

    print('train finished with epoch: %d / %d' % (e + 1, config.epoch), flush = True)
    print('best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' % (max_epoch, max_dev_f, max_test_f), flush = True)
    print(str(datetime.datetime.now()), flush = True)
