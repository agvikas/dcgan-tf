from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

class Model:

    def __init__(self, parameters):

        self.initialize = tf.random_normal_initializer(mean=0, stddev=0.02)
        '''self.initialize = tf.contrib.layers.xavier_initializer()'''
        self.decay = parameters['moving_average_decay']
        self.mode = parameters['mode']
        self.batch_size = parameters['batch_size']

    '''def generator(self, noise):

        with tf.variable_scope('generator'):
            shape = noise.get_shape()
            
            self.gen_dense1 = tf.layers.dense(noise, 4*4*512, kernel_initializer=self.initialize, name='gen_dense1')
            self.dense1_reshape = tf.reshape(self.gen_dense1, [-1, 4, 4, 512], name='gen_dense1_reshape')
            self.gen_bn1 = self._batch_normalization(self.dense1_reshape)
            self.gen_bn1_act = tf.nn.relu(self.gen_bn1, name='gen_bn1_act')
            
            self.gen_conv1 = self._conv_layer(self.gen_bn1_act, 3, 1, 256, 'gen_conv1', 'relu')
            self.deconv_2 = self._deconv_layer(self.gen_conv1, 2, 2, 256, 'deconv_2')
            self.gen_conv2 = self._conv_layer(self.deconv_2, 3, 1, 128, 'gen_conv2', 'relu')
            self.deconv_3 = self._deconv_layer(self.gen_conv2, 2, 2, 128, 'deconv_3')
            self.gen_conv3 = self._conv_layer(self.deconv_3, 3, 1, 64, 'gen_conv3', 'relu')
            self.deconv_4 = self._deconv_layer(self.gen_conv3, 2, 2, 64, 'deconv_4')
            self.gen_conv4 = self._conv_layer(self.deconv_4, 3, 1, 32, 'gen_conv4', 'relu')
            self.deconv_5 = self._deconv_layer(self.gen_conv4, 2, 2, 3, 'deconv_5')
            self.gen_output = self._conv_layer(self.deconv_5, 3, 1, 3, 'gen_output', 'tanh', bn=False) '''


    def generator(self, noise):

        with tf.variable_scope('generator'):
            shape = noise.get_shape()
            
            gen_dense1 = tf.layers.dense(noise, 4*4*512, kernel_initializer=self.initialize, name='gen_dense1')
            dense1_reshape = tf.reshape(gen_dense1, [-1, 4, 4, 512], name='gen_dense1_reshape')
            gen_bn1 = self._batch_normalization(dense1_reshape, name='gen_bn1')
            gen_bn1_act = tf.nn.relu(gen_bn1, name='gen_bn1_act')
            
            deconv_1 = self._deconv_layer(gen_bn1_act, 5, 2, 256, 'deconv_1')
            deconv_2 = self._deconv_layer(deconv_1, 5, 2, 128, 'deconv_2')
            deconv_3 = self._deconv_layer(deconv_2, 5, 2, 64, 'deconv_3')
            
            gen_output = self._deconv_layer(deconv_3, 5, 2, 3, 'gen_output', 'tanh', bn=False)
            '''shape = self.gen_output.get_shape()
            print("%d %d %d %d" %(shape[0].value, shape[1].value, shape[2].value, shape[3].value) )'''
            return gen_output


    def discriminator(self, dis_images, reuse=None):

        with tf.variable_scope('discriminator', reuse=reuse):

            dis_conv1 = self._conv_layer(dis_images, 3, 2, 64, 'dis_conv1')
            '''self.dis_pool1 = self._max_pool(self.dis_conv1, 'dis_pool1')'''

            dis_conv2 = self._conv_layer(dis_conv1, 3, 2, 128, 'dis_conv2')
            '''self.dis_pool2 = self._max_pool(self.dis_conv2, 'dis_pool2')'''

            dis_conv3 = self._conv_layer(dis_conv2, 3, 2, 256, 'dis_conv3')
            '''self.dis_pool3 = self._max_pool(self.dis_conv3, 'dis_pool3')'''

            dis_conv4 = self._conv_layer(dis_conv3, 3, 2, 512, 'dis_conv4')
            '''self.dis_pool4 = self._max_pool(self.dis_conv4, 'dis_pool4')'''

            flatten = tf.layers.flatten(dis_conv4, 'flat')

            '''self.dis_dense1 = tf.layers.dense(self.flatten, 512, kernel_initializer=self.initialize, name='dis_dense1')
            self.dis_bn1 = tf.layers.batch_normalization(self.dis_dense1, training=self.mode, name='dis_bn1')
            self.dis_dense1_act = tf.nn.leaky_relu(self.dis_dense1, name='dense1_act')'''

            '''self.dis_dense2 = tf.layers.dense(self.dis_dense1_act, 256, kernel_initializer=self.initialize, name='dis_dense2')
            self.dis_bn2 = tf.layers.batch_normalization(self.dis_dense2, training=self.mode, name='dis_bn2')
            self.dis_dense2_act = tf.nn.leaky_relu(self.dis_bn2, name='dense2_act')'''

            dis_logits = tf.layers.dense(flatten, 1, kernel_initializer=self.initialize, name='dis_logits')
            dis_output = tf.nn.sigmoid(dis_logits, name='dis_output')
            '''self.dis_softmax = tf.nn.softmax(self.dis_logits, -1, 'dis_softmax')'''
            return dis_output

    def _max_pool(self, feature_map, name):
        pool = tf.nn.max_pool(feature_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool

    def _conv_layer(self, input, filter_size, stride, out_channels, name, act='leaky_relu', bn=True):
        shape = input.get_shape()
        with tf.variable_scope(name):
            weights = tf.get_variable(name="filter", initializer=self.initialize,
                                      shape=[filter_size, filter_size, shape[-1], out_channels], dtype=tf.float32)
            bias = tf.get_variable(name="bias", initializer=tf.zeros_initializer, shape=[out_channels], dtype=tf.float32)
            conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')

            bias_add = tf.nn.bias_add(conv, bias)
            
            if bn:
                batch_norm = self._batch_normalization(bias_add, name+'_bn')
            else:
                batch_norm = bias_add
            
            print("activation of layer {} : {}".format(name, act))
            
            if act=='relu':
                activation = tf.nn.relu(batch_norm)
            elif act=='leaky_relu':
                activation = tf.nn.leaky_relu(batch_norm)
            elif act=='tanh':
                activation = tf.nn.tanh(batch_norm)
            else:
                activation = batch_norm
            return activation

    def _deconv_layer(self, input, filter_size, stride, out_channels, name, act='relu', bn=True):
        shape = input.get_shape()
        height = stride * shape[1].value
        width = stride * shape[2].value
        with tf.variable_scope(name):
            weights = tf.get_variable(name="filter", initializer=self.initialize,
                                      shape=[filter_size, filter_size, out_channels, shape[-1]], dtype=tf.float32)
            bias = tf.get_variable(name="bias", initializer=tf.zeros_initializer, shape=[out_channels], dtype=tf.float32)
            deconv = tf.nn.conv2d_transpose(input, weights, [tf.shape(input)[0], height, width, out_channels],
                                            [1, stride, stride, 1], padding='SAME')

            bias_add = tf.nn.bias_add(deconv, bias)

            if bn:
                batch_norm = self._batch_normalization(bias_add, name+'_bn')
            else:
                batch_norm = bias_add

            print("activation of layer {} : {}".format(name, act))
            
            if act=='relu':
                activation = tf.nn.relu(batch_norm)
            elif act=='leaky_relu':
                activation = tf.nn.leaky_relu(batch_norm)
            elif act=='tanh':
                activation = tf.nn.tanh(batch_norm)
            else:
                activation = batch_norm
            return activation

    def _batch_normalization(self, x, name):

        '''with tf.variable_scope(name):
            offset = tf.Variable(tf.zeros([x.get_shape()[-1]]), name='offset')
            scale = tf.Variable(tf.ones([x.get_shape()[-1]]), name='scale')
            pop_mean = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=False, name='pop_mean')
            pop_var = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=False, name='pop_var')

            if self.mode:
                mean, var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False, name='moments')
                train_mean = tf.assign(pop_mean, pop_mean * self.decay + mean * (1 - self.decay), name='mean_update')
                train_var = tf.assign(pop_var, pop_var * self.decay + mean * (1 - self.decay), name='var_update')
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=offset, scale=scale,
                                                     variance_epsilon=1e-4, name='bn')
            else:
                return tf.nn.batch_normalization(x=x, mean=pop_mean, variance=pop_var, offset=offset, scale=scale,
                                             variance_epsilon=1e-4, name='bn_inf')'''
        return tf.contrib.layers.batch_norm(x, decay=self.decay, scale=True, is_training=self.mode, scope=name)
