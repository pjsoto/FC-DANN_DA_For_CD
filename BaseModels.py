from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

import numpy as np
import pandas as pd
#import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.framework.python.ops import arg_scope

class MobileNet():
    def __init__(self, args, OS = 8):
        self.args = args
        self.output_stride = OS
    def build_Encoder_Layers(self, X, alpha = 1, bn_decay = 0.999, bn_eps = 1e-3, name = 'mobile_net'):
        chan_dim = -1
        Layers = []
        Layers.append(X)
        with tf.variable_scope(name):
            with tf.variable_scope('reduce_input_block'):
                first_block_filters = self.make_Divisible(32 * alpha, 8)
                Layers.append(tf.layers.conv2d(Layers[-1], first_block_filters, (3 , 3), strides = 2, use_bias = False, padding = 'SAME', activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer() ,name = 'convd_3x3'))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = 'bn_1'))
                Layers.append(tf.nn.relu(Layers[-1], name = 'relu6_1'))

            #Residul blocks
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=16, alpha=alpha, stride=1,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=1, block_id=0, skip_connection=False))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=24, alpha=alpha, stride=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=1, skip_connection=False))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=24, alpha=alpha, stride=1,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=2, skip_connection=True))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=32, alpha=alpha, stride=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=3, skip_connection=False))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=32, alpha=alpha, stride=1,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=4, skip_connection=True))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=32, alpha=alpha, stride=1,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=5, skip_connection=True))

            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=64, alpha=alpha, stride=1,bn_eps=bn_eps, bn_decay=bn_decay,  # 1!
                                    expansion=6, block_id=6, skip_connection=False))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=64, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=7, skip_connection=True))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=64, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=8, skip_connection=True))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=64, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=9, skip_connection=True))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=96, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=10, skip_connection=False))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=96, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=11, skip_connection=True))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=96, alpha=alpha, stride=1, rate=2,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=12, skip_connection=True))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=160, alpha=alpha, stride=1, rate=2, bn_eps=bn_eps, bn_decay=bn_decay,  # 1!
                                    expansion=6, block_id=13, skip_connection=False))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=160, alpha=alpha, stride=1, rate=4,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=14, skip_connection=True))
            Layers.append(self.inverted_Res_Block(Layers[-1], filters=160, alpha=alpha, stride=1, rate=4,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=15, skip_connection=True))

            Layers.append(self.inverted_Res_Block(Layers[-1], filters=320, alpha=alpha, stride=1, rate=4,bn_eps=bn_eps, bn_decay=bn_decay,
                                    expansion=6, block_id=16, skip_connection=False))
        return Layers

    def inverted_Res_Block(self, X, expansion, stride, alpha, filters, block_id, skip_connection, rate = 1, bn_eps = 1e-3, bn_decay = 0.999, chan_dim = -1):

        chan_size = X.shape[-1]
        pointwise_filters = self.make_Divisible(int(filters * alpha), 8)
        x = X
        with tf.variable_scope('inverse_Residual_Block' + str(block_id)):
            if block_id:
                with tf.variable_scope('expand'):
                    print('expand')
                    x = tf.layers.conv2d(x, expansion * chan_size, (1,1), use_bias=False,padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2d_1x1')
                    x = tf.layers.batch_normalization(x, axis=chan_dim, epsilon=bn_eps, momentum=bn_decay,name='bn')
                    x = tf.nn.relu(x, name='relu6')

            with tf.variable_scope('depthwise'):
                print('depthwise')
                depthwise_filter = tf.get_variable('filters', (3,3, x.shape[-1],1), tf.float32)
                x = tf.nn.depthwise_conv2d(x,filter=depthwise_filter, strides=[1,stride,stride,1], padding='SAME', name='depthwise_conv2d_3x3')
                x = tf.layers.batch_normalization(x, axis=chan_dim, epsilon=bn_eps, momentum=bn_decay,name='bn')
                x = tf.nn.relu(x, name='relu6')

            with tf.variable_scope('pointwise'):
                print('pointwise')
                x = tf.layers.conv2d(x, pointwise_filters, (1,1), use_bias=False,padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2d_1x1')
                x = tf.layers.batch_normalization(x, axis=chan_dim, epsilon=bn_eps, momentum=bn_decay,name='bn')
                x = tf.nn.relu(x, name='relu6')

            if skip_connection:
                x = tf.math.add(x, X, name='add_shortcut')


            return x

    def make_Divisible(self, v, divisor, min_value=None):

        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

class ResNetV2_34_18():
    def __init__(self, args):
        super(ResNetV2_34_18, self).__init__()
        self.args = args

    def build_Encoder_Layers(self, X, stages = (3, 4, 6, 3), filters = (64, 128, 256, 512), bn_eps = 2e-5, bn_decay = 0.9, name = "Resnet_V2_34"):

        Layers = []
        Layers.append(X)
        chan_dim = -1
        with tf.variable_scope(name):

            with tf.variable_scope('reduce_input_block'):
                # apply CONV => BN => ACT => POOL to reduce spatial size
                Layers.append(tf.layers.conv2d(Layers[-1], 64, 7,strides=(2, 2), use_bias=False, padding="same", activation=None, name="conv2d_7x7"))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=bn_eps, momentum=bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))
                Layers.append(tf.layers.max_pooling2d(Layers[-1], (3, 3), strides=(2, 2), padding="same", name = 'max_pooling2d'))

            for i in range(0, len(stages)):
                stride = (1 , 1) if i == 0 else (2 , 2)
                Layers.append(self.simple_Residual_Block(Layers[-1], filters[i], stride, chan_dim, red = True, bn_eps=bn_eps, bn_decay = bn_decay, name = "block" + str(i+1)+"/unit1"))

                for j in range(0, stages[i] - 1):
                    Layers.append(self.simple_Residual_Block(Layers[-1], filters[i], (1 , 1), chan_dim, red = True, bn_eps = bn_eps, bn_decay = bn_decay, name = "block" + str(i+1) + "/unit" + str(j+2)))


            with tf.variable_scope('last_bn_act_pooling'):
                # apply BN => ACT => POOL
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=bn_eps, momentum=bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))


            return Layers

    def residual_Block(self, X, filters = 64, chan_dim = -1, red = False, bn_eps = 2e-5, bn_decay = 0.9, name = 'block1'):

        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chan_dim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        reg: applies regularization strength for all CONV layers in the residual module
        bn_eps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bn_decay: controls the momentum for the moving average
        """
        Layers = []
        Layers.append(X)
        with tf.variable_scope(name):

            Layers.append(tf.layers.batch_normalization(Layers[-1], axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = 'bn_1'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_1'))
            Layers.append(tf.layers.conv2d(act1, int(filters * 0.25), (1, 1), use_bias = False, padding = 'SAME', activation = None, name = 'conv2d_1x1_1'))

            Layers.append(tf.layers.batch_normalization(Layers[-1], axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = 'bn_2'))
            Layers.append(tf.nn.relu(Layers[-1], name = 'relu_2'))
            Layers.append(tf.layers.conv2d(Layers[-1], int(filters * 0.25), (3, 3), strides = stride, padding = 'SAME', use_bias = False, activation = None, name = 'conv2d_3x3_2'))


            Layers.append(tf.layers.batch_normalization(Layers[-1], axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = 'bn_3'))
            Layers.append(tf.nn.relu(Layers[-1], name = 'relu_3'))
            Layers.append(tf.layers.conv2d(Layers[-1], filters, (1, 1), use_bias = False, padding = 'SAME', activation = None, name = 'conv2d_1x1_3'))

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if red:
                X = tf.layers.conv2d(Layers[-8], filters, (1, 1), strides=stride, use_bias=False, activation=None,name="reduce_conv2d_1x1_1")

            # add together the shortcut and the final CONV
            Layers.append(tf.math.add(Layers[-1], X, name='add_shortcut'))

            return Layers

    def simple_Residual_Block(self, X, filters = 64, stride = 2, chan_dim = -1, red = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        x = X
        with tf.variable_scope(name):
            # the first block of the ResNet module are the 1x1 CONVs
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = 'bn_1')
            x_act = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv2d(x_act, filters, (3, 3), use_bias = False, padding = "SAME",  activation = None, name = 'conv2d_1x1_1')


            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name='bn_2')
            x = tf.nn.relu(x, name = 'relu_2')
            x = tf.layers.conv2d(x, filters, (3, 3), strides=stride, padding="SAME", use_bias = False, activation = None, name='conv2d_3x3_2')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if red:
                X = tf.layers.conv2d(x_act, filters, (1, 1), strides = stride, use_bias = False, activation = None, name="reduce_conv2d_1x1")

            # add together the shortcut and the final CONV
            x = tf.math.add(x, X, name='add_shortcut')

        # return the addition as the output of the ResNet module
        return x

class Xception():
    def __init__(self, args):
        self.args = args

    def build_Encoder_Layers(self, X, name = 'xception'):
        Layers = []
        Layers.append(X)

        with tf.variable_scope(name):
            Layers.append(self.general_conv2d(Layers[-1], 32, 3, stride=2, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_1'))
            Layers.append(self.general_conv2d(Layers[-1], 64, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_2'))
            tensor = Layers[-1]

            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_3'))
            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_4'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_1'))


            tensor = self.general_conv2d(tensor, 128, 1, stride=2, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_5')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_1'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_1'))

            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_6'))
            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_7'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_1'))


            tensor = self.general_conv2d(tensor, 256, 1, stride=2, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_8')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_2'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_2'))


            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_9'))
            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_10'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_2'))


            tensor = self.general_conv2d(tensor, 728, 1, stride=2, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_11')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_3'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_3'))

            #Middle flow
            for i in range(8):
                tensor = Layers[-1]
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_1' + str(i)))
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_2' + str(i)))
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_3' + str(i)))
                Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_4' + str(i)))

            Layers.append(tf.nn.relu(Layers[-1], name='relu_4'))
            #Exit flow
            tensor = self.general_conv2d(Layers[-1], 1024, 1, stride=1, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_13')

            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_14'))
            Layers.append(self.general_conv2d(Layers[-1], 1024, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_15'))
            #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_3'))

            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_4'))

            Layers.append(self.general_conv2d(Layers[-1], 1536, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_16'))
            Layers.append(self.general_conv2d(Layers[-1], 2048, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_17'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, conv_type = 'conv', stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            if conv_type == 'conv':
                conv = tf.layers.conv2d(input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            if conv_type == 'dep_conv':
                conv = tf.contrib.layers.separable_conv2d(input_data, filters, kernel_size, 1, stride, padding, activation_fn = None, weights_initializer = tf.contrib.layers.xavier_initializer())

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv
