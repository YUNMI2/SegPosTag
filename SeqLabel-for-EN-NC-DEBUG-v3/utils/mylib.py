def is_start_label(label):
    return label[0] in ["B", "b", "S", "s", "O", "o"]

def is_continue_label(label, startLabel, distance):
    if distance == 0:
        return True
    elif distance != 0 and is_start_label(label):
        return False
    elif startLabel[0] in ["s", "S", "O", "o"]:
        return False
    elif startLabel[1:] != label[1:]:
        return False
    return True


def writeConll(fileName, wordSeq, predictSeq, targetSeq):
    if not fileName.strip():
        return 
    print("\nStart Writing file", end="", flush=True)
    i = 0
    with open(fileName, "w", encoding="utf-8") as fw:
        for word, predict, target in zip(wordSeq, predictSeq, targetSeq):
            assert word.__len__() == predict.__len__() == target.__len__()
            for x, y, z in zip(word, predict, target):
                fw.write(x + "\t" + y + "\t" + z + "\n")
            fw.write("\n") 

            i += 1
            if i % (predictSeq.__len__()//10) == 0:
                print(".", end="", flush=True)
    print("\nFinish Writing file!\n", flush=True)
