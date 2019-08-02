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
