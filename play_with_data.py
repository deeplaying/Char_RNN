import random
import os

window = 50
overlap = 25
filename = "sample_data.txt"

vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " '\"abcdefghijklmnopqrstuvwxyz{|}@#➡📈")


def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])


#todo: fix extractinig examples, right now it is ignoring the last part which does not fit overlap
def read_data(filename, vocab, window, overlap):
    lines = [line.strip() for line in open(filename, 'r').readlines()]
    while True: # this generator never goes off, control the steps in the model
        random.shuffle(lines)
        for text in lines:
            text = vocab_encode(text, vocab)
            for start in range(0, len(text) - window, overlap):
                chunk = text[start: start + window]
                chunk += [0] * (window - len(chunk)) # for padding, doing nothing so far
                # add X, Y data by shifting chunk, X is gonna be 0->49 and labels are from 1->50
                Y = chunk[1:]
                X = chunk[:-1]
                # print(len(labels)) 49
                # print(len(chunk[:-1]))  49
                yield X, Y


#
# stream = read_data(filename, vocab, window, overlap)
#
# X, y = next(stream)
#
# print(X)
# print(y)
#
# print("vocab size")
# print(len(vocab))