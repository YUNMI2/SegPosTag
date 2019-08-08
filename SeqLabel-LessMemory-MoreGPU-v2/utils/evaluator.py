import torch
import tqdm
from .prf import *


class Decoder(object):
    @staticmethod
    def viterbi(crf, emit_matrix):
        '''
        viterbi for one sentence
        '''
        length = emit_matrix.size(0)
        max_score = torch.zeros_like(emit_matrix)
        paths = torch.zeros_like(emit_matrix, dtype=torch.long)

        max_score[0] = emit_matrix[0] + crf.strans
        for i in range(1, length):
            emit_scores = emit_matrix[i]
            scores = emit_scores + crf.transitions + \
                max_score[i - 1].view(-1, 1).expand(-1, crf.labels_num)
            max_score[i], paths[i] = torch.max(scores, 0)
            #print(crf.transitions.size())
            #print(max_score[i - 1].view(-1, 1).expand(-1, crf.labels_num).size())
            #exit()

        max_score[-1] += crf.etrans
        prev = torch.argmax(max_score[-1])
        predict = [prev.item()]
        for i in range(length - 1, 0, -1):
            prev = paths[i][prev.item()]
            predict.insert(0, prev.item())
        return torch.tensor(predict)
    
    @staticmethod
    def viterbi_batch(crf, emits, masks):
        '''
        viterbi for sentences in batch
        '''
        emits = emits.transpose(0, 1)
        masks = masks.t()
        sen_len, batch_size, labels_num = emits.shape

        lens = masks.sum(dim=0)  # [batch_size]
        scores = torch.zeros_like(emits)  # [sen_len, batch_size, labels_num]
        paths = torch.zeros_like(emits, dtype=torch.long) # [sen_len, batch_size, labels_num]

        scores[0] = crf.strans + emits[0]  # [batch_size, labels_num]
        for i in range(1, sen_len):
            trans_i = crf.transitions.unsqueeze(0)  # [1, labels_num, labels_num]
            emit_i = emits[i].unsqueeze(1)  # [batch_size, 1, labels_num]
            score = scores[i - 1].unsqueeze(2)  # [batch_size, labels_num, 1]
            score_i = trans_i + emit_i + score  # [batch_size, labels_num, labels_num]
            scores[i], paths[i] = torch.max(score_i, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(scores[length - 1, i] + crf.etrans)
            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            predicts.append(torch.tensor(predict).flip(0))

        return predicts


class Evaluator(object):
    def __init__(self, name, vocab, config, net):
        assert name in ["Train", "Dev", "Test"]
        self.name = name 
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab
        self.config = config
        self.network = net
        self.wordSeq = []
        self.predictSeq = []
        self.targetSeq = []

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    def parse(self, batch, mask, out, targets):
        sen_lens = mask.sum(1)
        if self.config.multiGPU:
            predicts = Decoder.viterbi_batch(self.network.module.crf, out, mask)
        else:
            predicts = Decoder.viterbi_batch(self.network.crf, out, mask)
        
        targets = torch.split(targets[mask], sen_lens.tolist())
        words = torch.split(batch[0][mask], sen_lens.tolist())
        for word, predict, target in zip(words, predicts, targets):
            assert word.__len__() == predict.__len__() == target.__len__()
            word, predict, target = self.vocab.id2word(word.tolist()), self.vocab.id2label(predict.tolist()), self.vocab.id2label(target.tolist())
            self.wordSeq.append(word)
            self.predictSeq.append(predict)
            self.targetSeq.append(target)
            if not self.config.seg:
                correct_num = sum(x==y for x,y in zip(predict, target))
                self.correct_num += correct_num
                self.pred_num += len(predict)
                self.gold_num += len(target)
            else:
                correct_num, predict_num, target_num = PRF(predict, target).SegPos()
                self.correct_num += correct_num
                self.pred_num += predict_num
                self.gold_num += target_num


    def eval(self):
        precision = self.correct_num/self.pred_num
        recall = self.correct_num/self.gold_num
        f1 = 2*precision*recall/(precision+recall)
        self.clear_num()
        return precision, recall, f1




    def write(self, fileName):
        if not fileName.strip():
            return
        print("\nStart Writing file", end="", flush=True)
        i = 0
        with open(fileName, "w", encoding="utf-8") as fw:
            for word, predict, target in zip(self.wordSeq, self.predictSeq, self.targetSeq):
                assert word.__len__() == predict.__len__() == target.__len__()
                for x, y, z in zip(word, predict, target):
                    fw.write(x + "\t" + y + "\t" + z + "\n")
                fw.write("\n")

                i += 1
                if i % (self.predictSeq.__len__()//10) == 0:
                    print(".", end="", flush=True)
        print("\nFinish Writing file!\n", flush=True)    
        
    '''
    def eval(self, network, data_loader):
        network.eval()
        total_loss = 0.0
        total_num = 0
        
        wordSeq = []
        predictSeq = []
        targetSeq = [] 
        
        for batch in data_loader:
            batch_size = batch[0].size(0)
            total_num += batch_size

            mask, out, targets = network.module.forward_batch(batch)
            sen_lens = mask.sum(1)

            batch_loss = network.module.get_loss(out, targets, mask)
            total_loss += batch_loss * batch_size

            predicts = Decoder.viterbi_batch(network.module.crf, out, mask)
            targets = torch.split(targets[mask], sen_lens.tolist())
            words = torch.split(batch[0][mask], sen_lens.tolist())
            
            

            for word, predict, target in zip(words, predicts, targets):
                assert word.__len__() == predict.__len__() == target.__len__()
                word, predict, target = self.vocab.id2word(word.tolist()), self.vocab.id2label(predict.tolist()), self.vocab.id2label(target.tolist())
                wordSeq.append(word)
                predictSeq.append(predict)
                targetSeq.append(target)
                if not self.seg:
                    correct_num = sum(x==y for x,y in zip(predict, target))
                    self.correct_num += correct_num
                    self.pred_num += len(predict)
                    self.gold_num += len(target)
                else:
                    correct_num, predict_num, target_num = PRF(predict, target).SegPos()
                    self.correct_num += correct_num
                    self.pred_num += predict_num
                    self.gold_num += target_num
 

        precision = self.correct_num/self.pred_num
        recall = self.correct_num/self.gold_num
        f1 = 2*precision*recall/(precision+recall)

        self.clear_num()
        return total_loss/total_num, precision, recall, f1, wordSeq, predictSeq, targetSeq
    '''
