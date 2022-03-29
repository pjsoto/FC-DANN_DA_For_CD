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

class ResNetV1():
    def __init__(self, args):
        super(ResNetV1, self).__init__()
        self.args = args
        self.filters = (64, 128, 256, 512)
        self.bn_eps = 2e-5
        self.bn_decay = 0.9
        if '18' in self.args.backbone_name:
            self.stages = (2, 2, 2, 2)
        if ('34' in self.args.backbone_name) or ('50' in self.args.backbone_name):
            self.stages = (3, 4, 6, 3)
        if '101' in self.args.backbone_name:
            self.stages = (3, 4, 23, 3)
        if '152' in self.args.backbone_name:
            self.stages = (3, 8, 36, 3)

    def build_Encoder_Layers(self, X, name = "ResnetV1"):

        Layers = []
        Layers.append(X)
        chan_dim = -1
        with tf.variable_scope(name):

            with tf.variable_scope('reduce_input_block'):
                # apply CONV => BN => ACT => POOL to reduce spatial size
                Layers.append(self.ZeroPadding(Layers[-1], 7, "ZeroPadding_1"))
                Layers.append(tf.layers.conv2d(Layers[-1], 64, 7,strides=(2, 2), use_bias=False, padding="valid", activation=None, name="conv2d_7x7"))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=self.bn_eps, momentum=self.bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))
                Layers.append(self.ZeroPadding(Layers[-1], 3, "ZeroPadding_2"))
                Layers.append(tf.layers.max_pooling2d(Layers[-1], (3, 3), strides=(2, 2), padding="valid", name = 'max_pooling2d'))

            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    if j == 0 and i != 0:
                        strides = (2 , 2)
                    else:
                        strides = (1 , 1)
                    if ('18' in self.args.backbone_name) or ('34' in self.args.backbone_name):
                        if j == 0 and i == 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        elif j == 0 and i != 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                    else:
                        if j == 0:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
            return Layers

    def Residual_Block_1(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            x = tf.layers.conv2d(X, filters, (3 , 3), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_3x3_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name= name + 'relu_1')

            x = tf.layers.conv2d(x, filters, (3, 3), strides= (1, 1), use_bias = False, padding="same", activation = None, name = name + 'conv2d_3x3_2')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")
                X = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')

            # add together the shortcut and the final CONV
            x = tf.nn.relu(tf.math.add(x, X, name = name + 'add_shortcut'), name = name + 'relu_2')

            return x

    def Residual_Block_2(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):

            x = tf.layers.conv2d(X, filters, (1, 1), strides=stride, use_bias = False, padding="same", activation = None, name = name + 'conv2d_1x1_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name='relu_1')

            x = tf.layers.conv2d(x, filters, (3, 3), strides=(1 , 1), use_bias = False, padding="same", activation = None, name= name + 'conv2d_3x3_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name=name + 'bn_2')
            x = tf.nn.relu(x, name = 'relu_2')

            x = tf.layers.conv2d(x, 4 * filters, (1, 1), strides=(1 , 1), use_bias = False, padding="same", activation = None, name= name + 'conv2d_1x1_2')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')



            if shorcut:
                X = tf.layers.conv2d(X, 4 * filters, (1, 1), strides = stride, use_bias = False, activation = None, name=name + "shorcut_conv2d_1x1")
                X = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'shorcut_bn_4')

            # add together the shortcut and the final CONV
            x = tf.nn.relu(tf.math.add(x, X, name = name + 'add_shortcut'), name = name + 'relu_3')

        # return the addition as the output of the ResNet module
        return x

    def ZeroPadding(self, X, Kernel_size = 3, name = "ZeroPadding"):
        p = int((Kernel_size - 1)/2)
        output = tf.pad(X, [[0, 0], [p, p], [p, p], [0, 0]], mode='CONSTANT' , constant_values=0, name= name)
        return output

class ResNetV2():
    def __init__(self, args):
        super(ResNetV2, self).__init__()
        self.args = args
        self.filters = (64, 128, 256, 512)
        self.bn_eps = 2e-5
        self.bn_decay = 0.9
        if '18' in self.args.backbone_name:
            self.stages = (2, 2, 2, 2)
        if ('34' in self.args.backbone_name) or ('50' in self.args.backbone_name):
            self.stages = (3, 4, 6, 3)
        if '101' in self.args.backbone_name:
            self.stages = (3, 4, 23, 3)
        if '152' in self.args.backbone_name:
            self.stages = (3, 8, 36, 3)

    def build_Encoder_Layers(self, X, name = "ResnetV2"):
        Layers = []
        Layers.append(X)
        chan_dim = -1
        with tf.variable_scope(name):

            with tf.variable_scope('reduce_input_block'):
                # apply CONV => BN => ACT => POOL to reduce spatial size
                Layers.append(self.ZeroPadding(Layers[-1], 7, "ZeroPadding_1"))
                Layers.append(tf.layers.conv2d(Layers[-1], 64, 7,strides=(2, 2), use_bias=False, padding="valid", activation=None, name="conv2d_7x7"))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=self.bn_eps, momentum=self.bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))
                Layers.append(self.ZeroPadding(Layers[-1], 3, "ZeroPadding_2"))
                Layers.append(tf.layers.max_pooling2d(Layers[-1], (3, 3), strides=(2, 2), padding="valid", name = 'max_pooling2d'))

            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    if j == 0 and i != 0:
                        strides = (2 , 2)
                    else:
                        strides = (1 , 1)
                    if ('18' in self.args.backbone_name) or ('34' in self.args.backbone_name):
                        if j == 0 and i == 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        elif j == 0 and i != 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                    else:
                        if j == 0:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
            return Layers

    def Residual_Block_1(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            # the first block of the ResNet module are the 1x1 CONVs
            x = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name = name + 'relu_1')
            x = tf.layers.conv2d(x, filters, (3, 3), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_1x1_1')


            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, filters, (3, 3), strides=(1 , 1),  use_bias = False, padding="same", activation = None, name = name + 'conv2d_3x3_1')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")

            # add together the shortcut and the final CONV
            x = tf.math.add(x, X, name = name + 'add_shortcut')

            return x

    def Residual_Block_2(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            x = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name = name + 'relu_1')
            x = tf.layers.conv2d(x, filters, (1, 1), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_1x1_1')

            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, filters, (3, 3), strides = (1 , 1), use_bias = False, padding = "same", activation = None, name = name + 'conv2d_3x3_2')

            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, 4 * filters, (1, 1), strides = (1 , 1), use_bias = False, padding = "same", activation = None, name = name + 'conv2d_1x1_3')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, 4 * filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")

            # add together the shortcut and the final CONV
            x = tf.math.add(x, X, name = name + 'add_shortcut')

        # return the addition as the output of the ResNet module
        return x

    def ZeroPadding(self, X, Kernel_size = 3, name = "ZeroPadding"):
        p = int((Kernel_size - 1)/2)
        output = tf.pad(X, [[0, 0], [p, p], [p, p], [0, 0]], mode='CONSTANT' , constant_values=0, name= name)
        return output

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
            #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_2'))


            tensor = self.general_conv2d(tensor, 728, 1, stride=1, conv_type = 'conv',
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
