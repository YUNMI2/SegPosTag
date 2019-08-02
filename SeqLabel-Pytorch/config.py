class Config(object):
    #train_file = '../data/ctb5-train.conll'
    train_file = '../Data/google-text-normalization-data/TNFine-Data/Train90-Fine'
    dev_file = '../Data/google-text-normalization-data/TNFine-Data/Dev5-Fine'
    test_file = '../Data/google-text-normalization-data/TNFine-Data/Test5-Fine'
    embedding_file = '' #'../data/embedding/giga.100.txt'


class Char_LSTM_CRF_Config(Config):
    model = 'Char_LSTM_CRF'
    net_file = './save/char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 200
    layers = 2
    dropout = 0.55
    char_dim = 100
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    gpu = -1
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

    seg = True

    word_hidden = 300
    layers = 1
    dropout = 0.55
    word_dim = 100

    optimizer = 'adam'
    epoch = 30
    gpu = -1
    lr = 0.01
    batch_size = 256
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config, # python 可以将类名作为value值
    'bilstm_crf' : BiLSTM_CRF_Config,
}


