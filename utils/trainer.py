import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from .mylib import *
import tqdm
from .dataset import *
from .corpus import *

class Trainer(object):
    def __init__(self, network, config):
        self.network = network
        self.config = config
        # choose optimizer
        if config.optimizer == 'sgd':
            print('Using SGD optimizer...', flush = True)
            self.optimizer = optim.SGD(network.parameters(), lr=config.lr)
        elif config.optimizer == 'adam':
            print('Using Adam optimizer...', flush = True)
            self.optimizer = optim.Adam(network.parameters(), lr=config.lr)

    def lr_decay(self, optimizer, epoch, decay_rate, init_lr):
        lr = init_lr/(1+decay_rate*epoch)
        print("Learning rate is set as:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_files, test_data_loaders, evaluator):
        # record some parameters
        max_dev_f = 0
        max_test_f = 0
        max_epoch = 0
        patience = 0
      
        dev_loader, test_loader = test_data_loaders

        # begin to train
        print('start to train the model ', flush = True)
        for e in range(self.config.epoch):
            print('==============================Epoch<%d>==============================' % (e + 1), flush = True)
            self.network.train()
            time_start = datetime.datetime.now()

            if self.config.optimizer == 'sgd':
                self.lr_decay(self.optimizer, e, self.config.decay, self.config.lr)
                
            #for batch in tqdm.tqdm(train_loader):
            for one_train_file in train_files:
                train_sentences, train_labels = Corpus().getWordLabelSeq(one_train_file)
                train_data = process_data(evaluator.vocab, train_sentences, train_labels, max_word_len=20)
                train_loader = Data.DataLoader(
                    dataset=train_data,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn if not self.config.use_cuda else collate_fn_cuda
                )
                
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    mask, targets = batch[0].gt(0),batch[1]
                    total_length = batch[0].size()[1]
                    #print(batch[0].size()[1], flush=True)
                    out = self.network.forward(batch[0], total_length)
                    #mask, out, targets = self.network.module.forward_batch(batch)
                    loss = self.network.module.get_loss(out, targets, mask)
                   
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                    self.optimizer.step()
            '''
            with torch.no_grad():
                
                # 训练数据太多，只选择其中一个来代表测试Train的loss
                print("Computing Train Loss.........", flush = True)
                train_sentences, train_labels = Corpus().getWordLabelSeq(train_files[0])
                train_data = process_data(evaluator.vocab, train_sentences, train_labels, max_word_len=20)
                train_loader = Data.DataLoader(
                    dataset=train_data,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn if not self.config.use_cuda else collate_fn_cuda
                )
                train_loss, train_p, train_r, train_f, train_word, train_predict, train_target = evaluator.eval(self.network, train_loader)
                print('Train   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (train_loss, train_p, train_r, train_f), flush = True)
                
                print("Computing Dev Loss.........", flush = True)
                dev_loss, dev_p, dev_r, dev_f, dev_word, dev_predict, dev_target  = evaluator.eval(self.network, dev_loader)
                print('dev   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (dev_loss, dev_p, dev_r, dev_f), flush = True)

                print("Computing Test Loss.........", flush = True)
                test_loss, test_p, test_r, test_f, test_word, test_predict, test_target = evaluator.eval(self.network, test_loader)
                print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss, test_p, test_r, test_f), flush = True)
                
            # save the model when dev precision get better
            if dev_f > max_dev_f:
                max_dev_f = dev_f
                max_test_f = test_f
                max_epoch = e + 1
                patience = 0
                print('Ex best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' %(max_epoch, max_dev_f, max_test_f), flush = True)
                print('save the model...', flush = True)
                torch.save(self.network, self.config.net_file)
                if self.config.predictOut:
                #    writeConll(self.config.predict_train_file, train_word, train_predict, train_target)
                    writeConll(self.config.predict_dev_file, dev_word, dev_predict, dev_target)
                    writeConll(self.config.predict_test_file, test_word, test_predict, test_target)
            else:
                patience += 1

            time_end = datetime.datetime.now()
            print('iter executing time is ' + str(time_end - time_start) + '\n', flush = True)
            if patience > self.config.patience:
                break

        print('train finished with epoch: %d / %d' % (e + 1, self.config.epoch), flush = True)
        print('best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' %
            (max_epoch, max_dev_f, max_test_f), flush = True)
        print(str(datetime.datetime.now()), flush = True)
        '''
