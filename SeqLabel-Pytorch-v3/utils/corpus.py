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


