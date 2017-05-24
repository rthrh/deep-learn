from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
import tensorflowvisu
import math
import sys

import random

# import argparse


FLAGS = None
from matplotlib import pylab
from data import *

        

def main(_):
    # Import data
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    dat = Data()
    # data,labels = dat.prepare_data()
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 13])  ##total number of input data units
    W = tf.Variable(tf.zeros([13, 10]))   ## [x,y]  x-> input data size, y-> possible output values
    b = tf.Variable(tf.zeros([10]))       ## [x] -> possible output values
    y = tf.matmul(x, W) + b               ## biases 
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # tf.global_variables_initializer().run()
    
    
    # Train
    for epoch in range(10):
        print ("Epoch : ",epoch)
        dat.reset_batch_counter()
        for i in range(100):
               
        
        
            batch_xs, batch_ys = dat.get_next_batch()
            
            # shuffled_index = list(range(len(batch_xs)))
            # random.seed(4456)
            # random.shuffle(shuffled_index)
            # print (batch_xs.shape,batch_ys.shape)
            # batch_xs = np.array([batch_xs[i] for i in shuffled_index],'float32')
            # batch_ys = np.array([batch_ys[i] for i in shuffled_index],'float32')
            # print (batch_xs.shape,batch_ys.shape)
            # print (batch_xs)
            
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # print('Accuracy batch: ', accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}))
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_xs}))
        # print('Accuracy batch: ', accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}))  

        
        xxxx,yyyy = dat.get_whole_data()
        print('Accuracy whole: ', accuracy.eval(feed_dict={x: xxxx, y_: yyyy}))                                    
        print('Cross entropy: ', cross_entropy.eval(feed_dict={x: xxxx, y_: yyyy}))                                   
        print('Accuracybatch: ', accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}))                                    
                                          
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        # help='Directory for storing input data')
    # FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main)

  
  

