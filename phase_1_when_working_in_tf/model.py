import tensorflow as tf
import numpy as np
import os
import random
from play_with_data import *





class CharRnn(object):
    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def _import_data(self):
        # create the iterator
        iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.X, self.Y = iterator.get_next()

        self.init_iterator = iterator.make_initializer(dataset=self.dataset)


    def build_graph(self):
        self._import_data()

    def train(self, num_iterations):

        with tf.Session() as sess:

            #initialize the data
            sess.run(self.init_iterator)

            for i in range(num_iterations):
                try:
                    print(sess.run(self.X))
                    print(sess.run(self.Y))
                    print(sess.run(tf.shape(self.X)))
                    print(sess.run(tf.shape(self.Y)))
                except tf.errors.OutOfRangeError:
                    print("we got out of range error")
                    print("we gotta re initialize the dataset")