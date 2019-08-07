class Config(object):
    '''
    train_file = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/Train90-Fine.SegPos.conll'
    dev_file = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/Dev5-Fine.SegPos.conll'
    test_file = '/search/odin/zhuyun/Data/google-text-normalization-data/TNFine-Data/Test5-Fine.SegPos.conll'
    eval_file = '/search/odin/zhuyun/Data/PostData/EN_NUM/english_num2.addPUNCT.conll'
    '''
    embedding_file = '' #'../data/embedding/giga.100.txt'
    
    train_file = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-train.segpos.conll'
    #train_file = ""
    dev_file = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-dev.segpos.conll'
    test_file = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-test.segpos.conll'
    eval_file = '/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/origin/ctb5-test.segpos.conll'
    #train_files_hold = "/search/odin/zhuyun/Data/WordSeg/CTB5/SEGPOS/split/train/"
    train_files_hold = ""
    predict_train_file = './predict/ctb5-train-out.segpos.conll'
    predict_dev_file = './predict/ctb5-dev-out.segpos.conll'
    predict_test_file = './predict/ctb5-test-out.segpos.conll'
    predict_eval_file = './eval/PostEN.addPUNCT.predict.conll' 
    
class Char_LSTM_CRF_Config(Config):
    model = 'Char_LSTM_CRF'
    net_file = './save/char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    use_cuda = False
    multiGPU = False
    word_hidden = 300
    char_hidden = 200
    layers = 2
    dropout = 0.55
    char_dim = 100
    word_dim = 100

    predictOut = False
    
    optimizer = 'adam'
    epoch = 100
    gpu = "" 
    lr = 0.001
    batch_size = 5
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True
    seg = False


class BiLSTM_CRF_Config(Config):
    model = "BiLSTM_CRF"
    net_file = './save/bilstm_crf.pt'
    vocab_file = './save/vocab.pkl'
    
    use_cuda = False
    multiGPU = False
    seg = True

    word_hidden = 300
    layers = 1
    dropout = 0.55
    word_dim = 100

    #predictOut = True
    predictOut = False 
    
    optimizer = 'adam'
    epoch = 1000
    gpu = ""
    lr = 0.01
    batch_size = 64
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 100
    shuffle = True


config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config, # python 可以将类名作为value值
    'bilstm_crf' : BiLSTM_CRF_Config,
}


