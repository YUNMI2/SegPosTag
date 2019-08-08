'''
class Corpus(object):
    def __init__(self, filename=None):
        self.filename = filename
        self.sentence_num = 0
        self.word_num = 0
        self.word_seqs = []
        self.label_seqs = []
        sentence = []
        sequence = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                if line == '\n':
                    self.word_seqs.append(sentence)
                    self.label_seqs.append(sequence)
                    self.sentence_num += 1
                    sentence = []
                    sequence = []
                else:
                    conll = line.split()                    
                    sentence.append(conll[0])
                    sequence.append(conll[-1])
                    self.word_num += 1
                line = f.readline()
        print('%s : sentences:%d，words:%d' % (filename, self.sentence_num, self.word_num))

'''




class Corpus(object):
    def stat(self, files, corpus_type, getContent = False):
        assert corpus_type in ["Train", "Dev", "Test"]
        assert isinstance(files, str) or isinstance(files, list)
        files = files if isinstance(files, list) else [files]
        sentence_num, word_num = 0, 0
        word_freq = {}
        label_freq = {}
        for onefile in files:
            with open(onefile, "r", encoding="utf-8") as fo:
                line = fo.readline()
                while line:
                    if line.strip():
                        if corpus_type == "Train" and getContent:
                            [word, label] = line.strip().split("\t")
                            word_freq[word] = word_freq.get(word, 0) + 1
                            label_freq[label] = label_freq.get(label, 0) + 1
                        word_num += 1
                    else:
                        sentence_num += 1
                    line = fo.readline()
                        
        print('in %s : sentences:%d，words:%d' % (corpus_type, sentence_num, word_num), flush=True) 
        return word_freq, label_freq


    def getWordLabelSeq(self, fileName):
        sentence_seq = []
        label_seq = []
        with open(fileName, "r", encoding="utf-8") as fo:
            one_sentence_word = []
            one_sentence_label = []
            line = fo.readline()
            while line:
                 if line.strip():
                     one_sentence_word.append(line.strip().split("\t")[0])
                     one_sentence_label.append(line.strip().split("\t")[-1])
                 else:
                     assert one_sentence_word and one_sentence_label
                     sentence_seq.append(one_sentence_word)
                     label_seq.append(one_sentence_label)
                     one_sentence_word, one_sentence_label = [], []
                 line = fo.readline()
        return sentence_seq, label_seq
    
 
