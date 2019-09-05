import argparse
import torch
import torch.utils.data as Data
from config import *
from model.bilstm_crf import BiLSTM_CRF
from model.bilstm import BiLSTM
from utils import *
from utils.dataset import *
from utils.mylib import *
import os 
import torch.optim as optim
import datetime
import torch.nn as nn
import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import warnings
import json
warnings.filterwarnings('ignore')


def forward_batch(net, batch):
    word_idxs, label_idxs = batch
    mask = word_idxs.gt(0)
    max_length = batch[0].size()[1]
    out = net.forward(word_idxs, max_length)
    return mask, out, label_idxs 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--data', type=str, default="CTB5-POS", help="dataset to train")
    parser.add_argument('--gpu', type=str, default="SingleCPU", help='choose train mode, SingleCPU/SingleGPU/MultiGPU/DistGPU')
    parser.add_argument('--model', type=str, default="bilstm", help='choose model, bilstm/bilstm_crf')
    
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--thread', type=int, default=4, help='thread num')
    
    parser.add_argument('--rank', type=int, default=0, help='in dist GPU mode, rank of current process')
    args = parser.parse_args()
    print('parser:', flush = True)
    print(args, flush = True)
    print(flush=True)

    config = Config(data_config[args.data], gpu_config[args.gpu], model_config[args.model]).paraDict
    print("config:", flush = True)
    for k, v in config.items():
        print("%s =  %s"%(k,v), flush=True)   
    save_pkl(config, config["config_path"])
    print(flush=True) 

    # set para
    torch.set_num_threads(args.thread)
    torch.manual_seed(args.seed)
    print('CPU seed = %d' % torch.initial_seed(), flush = True)

    if config.get("use_cuda", False):
        torch.cuda.set_device(config["gpu_ids"][args.rank])
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print("GPU id = %d" % (config["gpu_ids"][0]), flush =True)
        print('GPU seed = %d' % torch.cuda.initial_seed(), flush = True)
    print(flush=True)
    
    # read training , dev and test file
    print('loading three datasets...', flush = True)
   
    # get all filenames 
    train_files = [config["train_path"].strip()] if os.path.isfile(config["train_path"].strip()) else loadAllFile(config["train_path"].strip())  
    train_word_freq, train_label_freq = Corpus().stat(train_files , "Train", True)

    dev_files = config["dev_path"].strip()
    dev_word_freq, dev_label_freq = Corpus().stat(dev_files, "Dev", False)

    test_files = config["test_path"].strip()
    test_word_freq, test_label_freq = Corpus().stat(test_files, "Test", False)
    
    # build, show and save vocab
    vocab = Vocab(train_word_freq, train_label_freq, min_freq=1)
    print('Words : %d, labels : %d'%(vocab.num_words, vocab.num_labels), flush = True)
    save_pkl(vocab, config["vocab_file"].strip())

    # clear Memory 
    del train_word_freq, train_label_freq
    del dev_word_freq, dev_label_freq
    del test_word_freq, test_label_freq

    # build natwork
    if args.model == "bilstm_crf":  
        net = BiLSTM_CRF(vocab.num_words,
                     config["word_dim"],
                     config["layers"],
                     config["word_hidden"],
                     vocab.num_labels,
                     config["dropout"],
        )  
    elif args.model == "bilstm":  
        net = BiLSTM(vocab.num_words,
                     config["word_dim"],
                     config["layers"],
                     config["word_hidden"],
                     vocab.num_labels,
                     config["dropout"],
        )
    print(net)

    # init optim 
    if config["optimizer"] == 'adam':
         print('Using Adam optimizer...', flush = True)
         optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    
    # if use GPU , move all needed tensors to CUDA
    if config.get("use_cuda", False) and not config.get("useMultiGPU", False) and not config.get("useDistGPU", False):
        net.cuda()
    
    elif config.get("useMultiGPU", False):
        net = torch.nn.DataParallel(net, device_ids=config["gpu_ids"])
        net.cuda()
    elif config.get("useDistGPU", False):
        torch.distributed.init_process_group(backend=config["backend"], init_method=config["share_method"] + config["ip_addr"] + ":" + config["ip_port"], rank=args.rank, world_size=config["world_size"])
        net.cuda()
        net = DistributedDataParallel(net, device_ids=[config["gpu_ids"][args.rank]], output_device=int(args.rank))



    # load and pack dev data  
    dev_sentences, dev_labels = Corpus().getWordLabelSeq(dev_files)
    dev_loader = Data.DataLoader(
        dataset=process_data(vocab, dev_sentences, dev_labels, max_word_len=20),
        batch_size=config["eval_batch"],
        shuffle=False,
        collate_fn=collate_fn if not config.get("use_cuda", False)  else collate_fn_cuda
    )
    del dev_sentences, dev_labels
 
    # load and pack test data  
    test_sentences, test_labels = Corpus().getWordLabelSeq(test_files)
    test_loader = Data.DataLoader(
        dataset=process_data(vocab, test_sentences, test_labels, max_word_len=20),
        batch_size=config["eval_batch"],
        shuffle=False,
        collate_fn=collate_fn if not config.get("use_cuda", False) else collate_fn_cuda
    )
    del test_sentences, test_labels

    print("Start to Train a model........", flush=True)

    # init some para in train 
    print("init ...", flush=True)
    max_dev_f, final_test_f = 0.0, 0.0
    max_epoch, patience = 0, 0

    print(flush=True)
    for e in range(config["epoch"]):
        print('--------------------------------------------Epoch<%d>-------------------------------------------- '%(e+1), flush=True)
        net.train()
        time_start = datetime.datetime.now()
        file_num = 0
        for train_file in train_files:
             file_start_time = datetime.datetime.now()
             file_num += 1
             print("[%s/%s]"%(file_num, train_files.__len__()), end=" ", flush=True)
             train_sentences, train_labels = Corpus().getWordLabelSeq(train_file)
             train_dataset = process_data(vocab, train_sentences, train_labels, max_word_len=20)
             if config.get("useDistGPU", False):
                 train_datasampler = DistributedSampler(train_dataset, num_replicas=int(config["world_size"]), rank=int(args.rank))
                 train_loader = Data.DataLoader(
                     dataset=train_dataset,
                     sampler=train_datasampler,
                     batch_size=config["batch_size"],
                     shuffle=False,
                     collate_fn=collate_fn if not config.get("use_cuda", False) else collate_fn_cuda
                 )
             else:
                 train_loader = Data.DataLoader(
                   dataset=train_dataset,
                   batch_size=config["batch_size"],
                   shuffle=config["shuffle"],
                   collate_fn=collate_fn if not config.get("use_cuda", False) else collate_fn_cuda
             )
             del train_sentences, train_labels
             
             #for batch in tqdm.tqdm(train_loader):
             for batch in train_loader:
                 optimizer.zero_grad()
                 mask, out, targets = forward_batch(net, batch)
                 if config.get("useMultiGPU", False) or config.get("useDistGPU",False) :
                     loss = net.module.get_loss(out, targets, mask)
                 else:
                     loss = net.get_loss(out, targets, mask)
                 loss.backward()
                 nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                 optimizer.step()
                 
             file_end_time = datetime.datetime.now()
             print(file_end_time - file_start_time, end="\t", flush=True)

        print("\nStart Test...........", flush=True)
        with torch.no_grad():
            net.eval()
            train_loss, dev_loss, test_loss = 0.0, 0.0, 0.0
            
            train_sentences, train_labels = Corpus().getWordLabelSeq(train_files[0])
            train_loader = Data.DataLoader(
                  dataset=process_data(vocab, train_sentences, train_labels, max_word_len=20),
                  batch_size=config["batch_size"],
                  shuffle=False,  
                  collate_fn=collate_fn if not config.get("use_cuda", False) else collate_fn_cuda
            )                   
            del train_sentences, train_labels

            print("Computing Train Loss.........", flush=True)
            train_evaler = Evaluator("Train", vocab, config, net)
            num_batch = 0
            for batch in train_loader:
                num_batch += 1
                mask, out, targets = forward_batch(net, batch)
                if config.get("useMultiGPU", False) or config.get("useDistGPU",False) :
                    train_loss += net.module.get_loss(out, targets, mask)  
                else:
                    train_loss += net.get_loss(out, targets, mask)  
                train_evaler.parse(batch, mask, out, targets)
            train_p, train_r, train_f = train_evaler.eval()
            print('Train   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (train_loss/num_batch, train_p, train_r, train_f), flush = True)
             
            print("Computing Dev Loss..........", flush=True)
            dev_evaler = Evaluator("Dev", vocab, config, net)
            num_batch = 0
            for batch in dev_loader:
                num_batch += 1
                mask, out, targets = forward_batch(net, batch)
                if config.get("useMultiGPU", False) or config.get("useDistGPU",False) :
                    dev_loss += net.module.get_loss(out, targets, mask)  
                else:
                    dev_loss += net.get_loss(out, targets, mask)  
                
                dev_evaler.parse(batch, mask, out, targets)
            dev_p, dev_r, dev_f = dev_evaler.eval()
            print('dev   : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (dev_loss/num_batch, dev_p, dev_r, dev_f), flush = True)
             
            print("Computing Test Loss..........", flush=True)
            test_evaler = Evaluator("Test", vocab, config, net)
            num_batch = 0
            for batch in test_loader:
                num_batch += 1
                mask, out, targets = forward_batch(net, batch)
                if config.get("useMultiGPU", False) or config.get("useDistGPU",False) :
                    test_loss += net.module.get_loss(out, targets, mask)  
                else:    
                    test_loss += net.get_loss(out, targets, mask)  
                test_evaler.parse(batch, mask, out, targets)
            test_p, test_r, test_f = test_evaler.eval()
            print('test  : loss = %.4f  precision = %.4f  recall = %.4f  f1 = %.4f' % (test_loss/num_batch, test_p, test_r, test_f), flush = True)

        # save the model when dev precision get better
        if dev_f > max_dev_f:
            max_dev_f = dev_f
            max_test_f = test_f
            max_epoch = e + 1
            patience = 0
            print('Ex best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' %(max_epoch, max_dev_f, max_test_f), flush = True)
            print('save the model...', flush = True)
            if config.get("useMultiGPU", False) or config.get("useDistGPU",False) :
                torch.save(net.module, config["net_file"] + "-epoch" + str(e))
            else:
                torch.save(net, config["net_file"] + "-epoch" + str(e))
            if config["savePredict"]:
                test_evaler.write(config["test_out_path"] + "-epoch" + str(e))
        else: 
            patience += 1
        del train_evaler, dev_evaler, test_evaler
        
        time_end = datetime.datetime.now()
        print('iter executing time is ' + str(time_end - time_start) + '\n', flush = True)
        if patience > config["patience"]:
            break

    if config.get("useDistGPU", False):
        torch.distributed.destroy_process_group()

    print('train finished with epoch: %d / %d' % (e + 1, config["epoch"]), flush = True)
    print('best epoch is epoch = %d ,the dev f1 = %.4f the test f1 = %.4f' % (max_epoch, max_dev_f, max_test_f), flush = True)
    print(str(datetime.datetime.now()), flush = True)

