from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#-----para resunet-----
import os
import sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
#--------------------



class Unet():
    def __init__(self, args):
        super(Unet, self).__init__()
        self.args = args

    def build_Unet_Arch(self, input_data, name="Unet_Arch"):
        with tf.variable_scope(name):
            # Encoder definition
            o_c1 = self.general_conv2d(input_data, self.args.base_number_of_features, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_1')
            o_mp1 = tf.layers.max_pooling2d(
                o_c1, 2, 2, name=name + '_maxpooling_1')
            o_c2 = self.general_conv2d(o_mp1, self.args.base_number_of_features * 2, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_2')
            o_mp2 = tf.layers.max_pooling2d(
                o_c2, 2, 2, name=name + '_maxpooling_2')
            o_c3 = self.general_conv2d(o_mp2, self.args.base_number_of_features * 4, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_3')
            o_mp3 = tf.layers.max_pooling2d(
                o_c3, 2, 2, name=name + '_maxpooling_3')
            o_c4 = self.general_conv2d(o_mp3, self.args.base_number_of_features * 8, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_4')
            o_mp4 = tf.layers.max_pooling2d(
                o_c4, 2, 2, name=name + '_maxpooling_4')
            o_c5 = self.general_conv2d(o_mp4, self.args.base_number_of_features * 16, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_5')
            o_mp5 = tf.layers.max_pooling2d(
                o_c5, 2, 2, name=name + '_maxpooling_5')
            o_c6 = self.general_conv2d(o_mp5, self.args.base_number_of_features * 16, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_6')

            # Decoder definition
            o_d1 = self.general_deconv2d(o_c6, self.args.base_number_of_features * 8, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_1')
            o_me1 = tf.concat([o_d1, o_c5], 3)  # Skip connection
            o_d2 = self.general_deconv2d(o_me1, self.args.base_number_of_features * 4, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_2')
            o_me2 = tf.concat([o_d2, o_c4], 3)  # Skip connection
            o_d3 = self.general_deconv2d(o_me2, self.args.base_number_of_features * 2, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_3')
            o_me3 = tf.concat([o_d3, o_c3], 3)  # Skip connection
            o_d4 = self.general_deconv2d(o_me3, self.args.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_4')
            o_me4 = tf.concat([o_d4, o_c2], 3)  # Skip connection

            o_d5 = self.general_deconv2d(o_me4, self.args.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_5')
            o_me5 = tf.concat([o_d5, o_c1], 3)  # Skip connection
            logits = tf.layers.conv2d(
                o_me5, self.args.num_classes, 1, 1, 'SAME', activation=None)
            prediction = tf.nn.softmax(logits, name=name + '_softmax')

            #RETURN DO BOTTLENECK MODIFICADO: ORIGINAL ERA O_C5
            return logits, prediction, o_c6

    def build_Unet_Encoder(self, X, name="Unet_Encoder_Arch"):
        Layers = []
        with tf.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(X)
                else:
                    Layers.append(self.general_conv2d(Layers[-1], self.args.base_number_of_features * (2**(i)), 3, stride=1,
                                                      padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_' + str(i)))
                    Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_' + str(i)))

            return Layers

    def build_Unet_Decoder(self, X, Encoder_Layers, name = "Unet_Decoder_Arch"):

        Layers = []
        with tf.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(X)
                else:
                    Layers.append(self.general_deconv2d(Layers[-1], self.args.base_number_of_features * (2**(self.args.encoder_blocks - i)), 3, stride=2,
                                                 padding='SAME', activation_function='relu', do_norm=True, name=name + '_deconv2d_' + str(i)))
                    if self.args.skip_connections:
                        d_shape = Layers[-1].get_shape().as_list()[1:]
                        for layer in Encoder_Layers:
                            e_shape = layer.get_shape().as_list()[1:]
                            if d_shape == e_shape:
                                Layers.append(tf.concat([Layers[-1], layer], 3))  # Skip connection
                                break

            Layers.append(tf.layers.conv2d(Layers[-1], self.args.num_classes, 1, 1, 'SAME', activation=None))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(
                input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

    def general_deconv2d(self, input_data, filters=64, kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="deconv2d"):
        with tf.variable_scope(name):
            deconv = tf.layers.conv2d_transpose(input_data, filters, kernel_size, (stride, stride), padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            if do_norm:
                deconv = tf.layers.batch_normalization(deconv, momentum=0.9)

            if activation_function == "relu":
                deconv = tf.nn.relu(deconv, name='relu')
            if activation_function == "leakyrelu":
                deconv = tf.nn.leaky_relu(deconv, alpha=relu_factor)
            if activation_function == "elu":
                deconv = tf.nn.elu(deconv, name='elu')

            return deconv

class Domain_Regressors():
    def __init__(self, args):
        super(Domain_Regressors, self).__init__()
        self.args = args
    #=============================GABRIEL: DOMAIN_CLASSIFIER=============================
    def build_Domain_Classifier_Arch(self, input_data, name="Domain_Classifier_Arch"):
        with tf.variable_scope(name):
            #Domain Classifier Definition: 2x (Fully_Connected_1024_units + ReLu) + Fully_Connected_1_unit + Logistic

            o_flatten = tf.layers.flatten(input_data)

            o_dense1 = self.general_dense(o_flatten, units=1024, activation_function="relu", name=name + '_dense1')
            o_dense2 = self.general_dense(o_dense1, units=1024, activation_function="relu", name=name + '_dense2')

            #SERÁ AQUI O ERRO??? ACTIVATION NONE NA VERDADE USA UMA ATIVAÇÃO LINEAR
            logits = tf.layers.dense(o_dense2, units=self.args.num_classes, activation=None)

            #ALÉM DISSO, AQUI EU USO UMA SOFTMAX, MAS EM MODELS, OUTRA SOFTMAX É USADA EM CIMA DESSE LOGITS (essa saída não é usada em models)
            prediction = tf.nn.softmax(logits, name=name + '_softmax')

            return logits, prediction

    #domain classifier -- Javier
    def D_4(self, X, reuse):
        def discrim_conv(name, X, out_channels, filtersize, stride=1, norm='', nonlin=True, init_stddev=-1):
            with tf.variable_scope(name) as scope:
                if init_stddev <= 0.0:
                    init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                else:
                    init = tf.truncated_normal_initializer(stddev=init_stddev)
                X = tf.layers.conv2d(X, out_channels, kernel_size=filtersize, strides=(stride, stride), padding="valid",
                                    kernel_initializer=init)
                if norm == 'I':
                    X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse, epsilon=0.001)
                elif norm == 'B':
                    X = tf.layers.batch_normalization(X, reuse=reuse, training=True)
                elif norm == 'G':
                    X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)
                if nonlin:
                    X = tf.nn.leaky_relu(X, 0.2)
                return X

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print('D in:', X.get_shape().as_list())

            X = self.conv_javier('DZ1', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv_javier('DZ2', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv_javier('DZ3', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv_javier('DZ4', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)

            print('D out before discrim:', X.get_shape().as_list())

            shape = X.get_shape().as_list()[1:3]

            print('D Tensor image shape:', shape)

            X = discrim_conv('d_out', X, 2, shape, norm=False, nonlin=False, init_stddev=0.02)

            print('D out:', X.get_shape().as_list())


            return X

    def conv_javier(self, id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0, dilation=1):

        assert padding in ["SAME", "VALID", "REFLECT", "PARTIAL"], 'valid paddings: "SAME", "VALID", "REFLECT", "PARTIAL"'
        if type(size) == int: size = [size, size]
        if init_stddev <= 0.0:
            init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        else:
            init = tf.truncated_normal_initializer(stddev=init_stddev)

        if padding == "PARTIAL":
            with tf.variable_scope('mask'):
                _, h, w, _ = input.get_shape().as_list()

                slide_window = size[0] * size[1]
                mask = tf.ones(shape=[1, h, w, 1])
                update_mask = tf.layers.conv2d(mask, filters=1, dilation_rate=(dilation, dilation), name='mask' + id,
                                            kernel_size=size, kernel_initializer=tf.constant_initializer(1.0),
                                            strides=stride, padding="SAME", use_bias=False, trainable=False)
                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('parconv'):
                x = tf.layers.conv2d(input, filters=channels, name='conv' + id, kernel_size=size, kernel_initializer=init,
                                    strides=stride, padding="SAME", use_bias=False)
                x = x * mask_ratio
                if use_bias:
                    bias = tf.get_variable("bias" + id, [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)
                return x * update_mask

        if padding == "REFLECT":
            assert size[0] % 2 == 1 and size[1] % 2 == 1, "REFLECTION PAD ONLY WORKING FOR ODD FILTER SIZE.. " + str(size)
            pad_x = size[0] // 2
            pad_y = size[1] // 2
            input = tf.pad(input, [[0, 0], [pad_x, pad_x], [pad_y, pad_y], [0, 0]], "REFLECT")
            padding = "VALID"

        return tf.layers.conv2d(input, channels, kernel_size=size, strides=[stride, stride],
                                padding=padding, kernel_initializer=init, name='conv' + id,
                                use_bias=use_bias, dilation_rate=(dilation, dilation))

    def general_dense(self, input_data, units=1024, activation_function="relu", use_bias=True, kernel_initializer=None,
                      bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,bias_regularizer=None, activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None, trainable=True, name='dense'):

        with tf.variable_scope(name):
            dense = tf.layers.dense(input_data, units, activation=None)

#            NÃO SEI SE É NECESSÁRIO COLOCAR O BATCH_NORM
#            if do_norm:
#                dense = tf.layers.batch_normalization(dense, momentum=0.9)

            if activation_function == "relu":
                dense = tf.nn.relu(dense, name='relu')
            if activation_function == "leakyrelu":
                dense = tf.nn.leaky_relu(dense, alpha=relu_factor)
            if activation_function == "elu":
                dense = tf.nn.elu(dense, name='elu')

            return dense

class SegNet():
    def __init__(self, args):
        super(SegNet, self).__init__()
        self.args = args

    def n_enc_block(self, inputs, n, k, name):
        h = inputs
        with tf.variable_scope(name):
            for i in range(n):
                h = tf.layers.conv2d(h, k, 3, 1, padding="SAME", kernel_initializer = tf.initializers.random_uniform(), activation=None)
                h = tf.layers.batch_normalization(h, momentum=0.9)
                h = tf.nn.relu(h, name='relu')

            h = tf.layers.max_pooling2d(h, 2, 2, name=name + '_maxpooling_' + str(i))
        return h

    def n_dec_block(self, inputs, n, k, name):
        h = inputs
        with tf.variable_scope(name):
            h = tf.keras.layers.UpSampling2D((2,2), name='unpool')(h)
            for i in range(n):
                if i == n:
                    h = tf.layers.conv2d(h, k/2, 3, 1, padding="SAME", activation=None)
                else:
                    h = tf.layers.conv2d(h, k, 3, 1, padding="SAME", activation=None)

                h = tf.layers.batch_normalization(h, momentum=0.9)
                h = tf.nn.relu(h, name='relu')
        return h

    def Build_SegNet_Encoder(self, input_data, ns, ks, name='SegNet_encoder'):
        Layers = []
        with tf.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(input_data)
                else:
                    Layers.append(self.n_enc_block(Layers[-1], n = ns[i - 1], k = ks[i - 1], name = 'eblock_' + str(i)))

            return Layers

    def Build_SegNet_Decoder(self, input_data, ns, ks, name = 'SegNet_decoder'):
        Layers = []
        with tf.variable_scope(name):
            for i in range(self.args.encoder_blocks + 1):
                if i == 0:
                    Layers.append(input_data)
                else:
                    Layers.append(self.n_dec_block(Layers[-1], n = ns[i - 1], k = ks[i - 1], name = 'dblock_' + str(i)))

            Layers.append(tf.layers.conv2d(Layers[-1], self.args.num_classes, 1, 1, 'SAME', activation=None))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))
            return Layers
