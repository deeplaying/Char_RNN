import tensorflow as tf
import numpy as np
import os
import random
from play_with_data import *
from model import *


WINDOW = 50
OVERLAP = 25
FILENAME = "sample_data.txt"
NUM_ITERATIONS = 10
VOCAB = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " '\"abcdefghijklmnopqrstuvwxyz{|}@#➡📈")

BATCH_SIZE = 64
NUM_STEPS = 49 #RNN UNROLLED

def gen():
    #todo turn this into args
    yield from read_data(FILENAME, VOCAB, WINDOW, OVERLAP)


def build_dataset(batch_size, num_steps):
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.int32, tf.int32),
                                   output_shapes=(tf.TensorShape([num_steps]), tf.TensorShape([num_steps])))

    dataset = dataset.batch(batch_size)
    return dataset



if __name__ == "__main__":

    dataset = build_dataset(BATCH_SIZE, NUM_STEPS)

    model = CharRnn(dataset)
    model.build_graph()
    model.train(NUM_ITERATIONS)