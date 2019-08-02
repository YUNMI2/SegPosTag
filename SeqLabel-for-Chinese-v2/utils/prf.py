from mylib import *

class PRF:
    def __init__(self, predictLabels, targetLabels):
        self.predict = predictLabels
        self.target = targetLabels

    def SegPos(self):
        target_seq = []
        idx = 0
        while idx < self.target.__len__():
            if is_start_label(self.target[idx]):
                idy, endpos = idx, -1
                while idy < self.target.__len__():
                    if not is_continue_label(self.target[idy], self.target[idx], idy-idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    idy += 1

                target_seq.append("[" + str(idx) + "," + str(endpos) + "]:"  + self.target[idx] + "~" +self.target[endpos])
                idx = endpos
            idx += 1

        predict_seq = []
        idx = 0
        while idx < self.predict.__len__():
            if is_start_label(self.predict[idx]):
                idy, endpos = idx, -1
                while idy < self.predict.__len__():
                    if not is_continue_label(self.predict[idy], self.predict[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    idy += 1

                predict_seq.append("[" + str(idx) + "," + str(endpos)+ "]:" + self.predict[idx] + "~" +self.predict[endpos])
                idx = endpos
            idx += 1

        # print(predict_seq)
        # print()
        # print(target_seq)
        # print([x for x in predict_seq if x in target_seq].__len__())
        # print(predict_seq.__len__())
        # print(target_seq.__len__())

        return [x for x in predict_seq if x in target_seq].__len__(), predict_seq.__len__(), target_seq.__len__()






if __name__ == "__main__":
    test = PRF(["B-DATE", "E-DATE", "S-DATE", "O", "B-DATE", "M-DATE", "E-DATE"],
               ["B-DATE", "E-DATE", "S-DATE", "S-DATE", "B-DATE", "M-DATE", "E-DATE"])

    test.SegPos()







