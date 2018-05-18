import tensorflow as tf
import numpy as np
import os
import random
from play_with_data import *
import utils

class CharRnn(object):
    def __init__(self, dataset, vocab, batch_size, time_series_length, cells_dims):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size
        self.time_series_length = time_series_length
        self.cells_dims = cells_dims # list of numbers
        pass

    def _measure_sequence_length(self, x):
        # version 1, for now will just output array of batch size containing one element which is the sequence length of rnn (49) remember (50-1 because input output generation)
        return [self.time_series_length] * self.batch_size

    def _import_data(self):
        with tf.name_scope("data"):
            # create the iterator
            iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
            self.X, self.Y = iterator.get_next()

            self.init_iterator = iterator.make_initializer(dataset=self.dataset)



    def _create_inference(self):
        with tf.name_scope("inference"):
            # current model does not support padding so will ignore that for now
            # 1- reshape input with one hot encoding to get 3d (batch, time_series, data_dim)  -> [64 49 86]
            with tf.name_scope("one_hot"):
                self.X_one_hot = tf.one_hot(self.X, len(self.vocab))
                self.Y_one_hot = tf.one_hot(self.Y, len(self.vocab))

            # 2- build the RNN architecture todo: play with to make it multi layer
            # version 1, just 1 lstm
            with tf.name_scope("RNN"):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.cells_dims[0], state_is_tuple=True)
                self.rnn_outputs, self.rnn_states = tf.nn.dynamic_rnn(
                    inputs=self.X_one_hot,
                    cell=cell,
                    dtype=tf.float32,
                    sequence_length=self._measure_sequence_length(None))

            # 3- GET final outputs by using dense layer
            with tf.name_scope("final_layer"):
                self.outputs = tf.layers.dense(inputs=self.rnn_outputs, units=len(self.vocab))


    def build_graph(self):
        self._import_data()
        self._create_inference()

    def train(self, num_iterations):
        saver = tf.train.Saver()
        variables_init_op = tf.global_variables_initializer()


        with tf.Session() as sess:
            # initialize variables
            sess.run(variables_init_op)
            #initialize the data
            sess.run(self.init_iterator)

            for i in range(num_iterations):
                try:
                    print(sess.run(self.X_one_hot))
                    print(sess.run(self.Y_one_hot))
                    print(sess.run(tf.shape(self.X_one_hot)))
                    print(sess.run(tf.shape(self.Y_one_hot)))

                    print("finaly layer")
                    print(sess.run(self.outputs))
                    print(sess.run(tf.shape(self.outputs)))

                except tf.errors.OutOfRangeError:
                    print("we got out of range error")
                    print("we gotta re initialize the dataset")