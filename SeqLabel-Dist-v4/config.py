class Google_Convert_Config:
    train_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v3/train/"
    dev_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v3/dev/Dev5.extract-short30-replaceCARDINAL-of-convert-split-convert"
    test_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v3/test/Test5.extract-short30-replaceCARDINAL-of-convert-split-convert"
    seg = True

class Google_Convert_v4_Config:
    train_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v4/convert-train-v4/"
    dev_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v4/convert-test-v4/Dev5.extract-short30-replaceCARDINAL-of-convert-v3"
    test_path = "/search/odin/zhuyun/Data/google-text-normalization-data/convert-v4/convert-test-v4/Test5.extract-short30-replaceCARDINAL-of-convert-v3"
    seg = True

class Google_TN_Data_Config:
    train_path = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/origin/short/train/'
    dev_path = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/origin/short/Dev5.short30'
    test_path = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/origin/short/Test5.short30'
    seg = True

class CTB5_POS_Data_Config:
    train_path = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/split/train/'
    dev_path = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-dev.segpos.conll'
    test_path = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-test.segpos.conll'
    seg = True

class SingleCPU_Config:
    use_cuda = False

class SingleGPU_Config:
    use_cuda = True
    useMultiGPU = False
    useDistGPU = False
    gpu_ids = [0]

class MultiGPU_Config:
    use_cuda = True
    useMultiGPU = True
    useDistGPU = False
    gpu_ids = [0, 1]

class DistGPU_Config:
    use_cuda = True
    useMultiGPU = False
    useDistGPU = True
    world_size = 4
    gpu_ids = [4, 5,6,7]
    backend = "nccl"
    share_method = "tcp://"
    ip_addr = "127.0.0.1"
    ip_port = "23451"

class BiLSTM_CRF_Config:
    model = 'BiLSTM_CRF'

    saveModel = True
    net_file = './save/bilstm_crf.pt'
    vocab_file = './save/vocab_crf.pkl'
    
    savePredict = True
    train_out_path = ""
    dev_out_path = ""
    test_out_path = "./predict/google.test.bilstm_crf.out"

    use_crf = True
  
    word_hidden = 300
    char_hidden = 200
    layers = 1
    dropout = 0.55
    char_dim = 100
    word_dim = 100
    
    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 5
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True

    config_path = "./conf/bilstm_crf.conf"

class BiLSTM_Config:
    model = 'BiLSTM'

    saveModel = True
    net_file = './save/bilstm.google_v4.pt'
    vocab_file = './save/vocab.google_v4.pkl'
    
    savePredict = True
    train_out_path = ""
    dev_out_path = ""
    test_out_path = "./predict/google_v4.test.bilstm.out" 

    use_crf = False
  
    word_hidden = 300
    char_hidden = 200
    layers = 1
    dropout = 0.55
    char_dim = 100
    word_dim = 100
    
    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 256
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True
    
    config_path = "./conf/google_v4.bilstm.conf"


class Config:
    def __init__(self, data, gpu, model):
        self.paraDict = dict() 
        for k,v in vars(data).items():
            if not (k.startswith("__") and k.endswith("__")):
                self.paraDict[k] = v 
        for k,v in vars(gpu).items():
            if not (k.startswith("__") and k.endswith("__")):
                self.paraDict[k] = v
        for k,v in vars(model).items():
            if not (k.startswith("__") and k.endswith("__")):
                self.paraDict[k] = v

    def show(self):
        for k,v in self.paraDict.items():
            print("%s = %s"%(k, v)) 

data_config = {
    "Google_v4"     : Google_Convert_v4_Config,
    "Google_v2"     : Google_Convert_Config,
    "Google"        : Google_TN_Data_Config,
    "CTB5-POS"      : CTB5_POS_Data_Config,

}

gpu_config = {
    'SingleCPU'     : SingleCPU_Config,
    'SingleGPU'     : SingleGPU_Config,
    'MultiGPU'      : MultiGPU_Config,
    'DistGPU'       : DistGPU_Config,
}


model_config = {
    'bilstm_crf'    : BiLSTM_CRF_Config,
    'bilstm'        : BiLSTM_Config,
}


if __name__ == "__main__":
    Config(data_config["Google"], gpu_config["SingleCPU"], model_config["bilstm"]).show()
	
