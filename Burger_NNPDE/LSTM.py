# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:44:38 2019

@author: Administrator
"""
import numpy as np
import tensorflow as tf

tf.set_random_seed(1)   # set random seed

batch_size=100
num_steps=500
num_classes=41
state_size=20
learning_rate=0.01

with tf.variable_scope('foo',reuse=tf.AUTO_REUSE):
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    
    init_state = tf.zeros([batch_size, state_size])
    
    rnn_inputs = tf.one_hot(x, num_classes)
    print(rnn_inputs.shape)
    #注意这里去掉了这行代码，因为我们不需要将其表示成列表的形式在使用循环去做。
    #rnn_inputs = tf.unstack(x_one_hot, axis=1)
    cell = tf.contrib.rnn.BasicRNNCell(state_size)
    #使用dynamic_rnn函数，动态构建RNN模型
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    
    print(rnn_outputs.shape,final_state.shape)
    
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.reshape(
                tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
                [batch_size, num_steps, num_classes])
    predictions = tf.nn.softmax(logits)
    
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
         
     