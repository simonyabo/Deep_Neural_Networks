#! /usr/bin/env python3

"""
@author: Simon Haile
This program creates a tensorflow graph with varying configurations.
Our program support both classification and regression problems.
Users can specify the number of hidden layers in their network as
well as the shape of the hidden layers. Users can choose to use minibatching,
which optimizer to use, learning rate, and various other variables.
For usage, run with the -h flag.
Example command:
python3 dnn.py train_x.txt train_y.txt dev_x.txt dev_y.txt 100 0.01 15 R sig grad 0.025
python3 dnn.py train_x.txt train_y.txt dev_x.txt dev_y.txt 100 0.01 15 R sig grad 0.025 -num_classes 10 -mb 1000 -nlayers 100
python3 dnn.py -v train_x.txt train_y.txt dev_x.txt dev_y.txt 100 0.01 15 R sig grad 0.025 -num_classes 10
"""

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import sys
import argparse
import numpy as np

np.set_printoptions(threshold=np.nan)

def build_graph(l0, y, out_dim, args):

    L = args.num_hidden_units
    C = args.num_classes
    # Determine activation function
    activation = args.hidden_unit_activation
    if activation == "sig":
        act = tf.sigmoid
    elif activation == "tanh":
        act = tf.tanh
    else:
        act = tf.nn.relu
    k_init = tf.random_uniform_initializer(minval=(args.init_range * -1), maxval= args.init_range)
    # Create arbitrarily deep neural network
    nlayers = args.nlayers
    for i in range(nlayers):
        #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
    # Determine which output configuration is
    # needed based on problem type
    if args.problem_mode == "C":
        #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output, name="cost")
        #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
    elif args.problem_mode == "R":
        output = tf.layers.dense(l0, out_dim, activation=None, use_bias=True, kernel_initializer=k_init, bias_initializer=k_init)
        #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
    return performance, cost

def parse_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="Verbose mode: Print train and dev performance for minibatches", action='store_true')
    #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
    parser.add_argument("-mb", type=int, help="Minibatch size", required=False)
    parser.add_argument("-nlayers", type=int, help="Number of hidden layers in your neural network", required=False, default=1)
    args = parser.parse_args()

    if args.problem_mode == "C" and args.num_classes is None:
        print("Number of classes must be specified for Classification.")
        exit()
    return args

def main():
    args = parse_arguments()
    # Load in train and dev data files
    t_in = np.loadtxt(args.train_feat)
    t_t = np.loadtxt(args.train_target)
    d_in = np.loadtxt(args.dev_feat)
    d_t = np.loadtxt(args.dev_target)
    # Handle if the input is scalars
    if len(t_in.shape) > 1:
        D = t_in.shape[1]
    else:
        D = 1
    # Define X and Y placeholders.
    # Handles if Y output is non-scalar
    x = tf.placeholder(shape=[None, D], dtype=tf.float32, name="inputs")
    if len(t_t.shape) > 1:
        out_dim = t_t.shape[1]
        y = tf.placeholder(shape=[None, out_dim], dtype=tf.float32, name="targets")
    else:
        out_dim = 1
        y = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")
    performance, cost = build_graph(x, y, out_dim, args)
    # Determine which optimizer to use
    opt = args.optimizer
    lr = args.learning_rate
    if opt == "adam":
        train_step = tf.train.AdamOptimizer(lr).minimize(cost)
    elif opt == "momentum":
        train_step = tf.train.MomentumOptimizer(lr, 0.5).minimize(cost)
    else:
        train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    init = tf.global_variables_initializer()
    # Run graph
    with tf.Session() as sess:
        sess.run(init)
        # Determine minibatch size
        if args.mb is None or args.mb == 0:
            num_batches = 1
        else:
            num_batches = int(len(t_in) / args.mb)
        epoch = args.epochs
        for i in range(epoch):
            best_obj = 0
            # Shuffle training data before batching
            permutation = np.random.permutation(len(t_in))
            shuffled_in = t_in[permutation]
            #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
            # Train all minibatches
            # If in verbose mode, print performance
            for j in range(num_batches):
                x_batch = input_batches[j]
                #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
                avg_obj_val += obj_val / num_batches
                if args.v:
                    #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
                    if#CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE:
                        print("Update", '%06d' % (j + 1), "train=", "{:.3F}".format(obj_val), "dev=", "{:.3F}".format(dev_obj_val[0]), file=sys.stderr)
                        best_obj =dev_obj_val[0]
            # Print epoch performance
            #CODE REMOVED TO MAINTAIN PRIVACY AS MUCH AS POSSIBLE
            print("Epoch", '%03d' % (i + 1), "train=", "{:.3F}".format(avg_obj_val), "dev=", "{:.3F}".format(dev_obj_val[0]), file=sys.stderr)
if __name__ == "__main__":
    main()
