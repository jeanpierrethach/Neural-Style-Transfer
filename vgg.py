import tensorflow as tf
import numpy as np
import scipy.io

class VGG19:

    VGG19_LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    def __init__(self, model_weights, pooling_type, verbose):
        self.model_weights = model_weights
        self.pooling_type = pooling_type
        self.verbose = verbose

    def _conv_layer(self, layer_name, layer_input, W):
        conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
        if self.verbose: 
            print('--{} | shape={} | weights_shape={}'.format(layer_name, conv.get_shape(), W.get_shape()))
        return conv

    def _relu_layer(self, layer_name, layer_input, b):
        relu = tf.nn.relu(layer_input + b)
        if self.verbose:
            print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), b.get_shape()))
        return relu

    def _pool_layer(self, layer_name, layer_input):
        if self.pooling_type == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif self.pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if self.verbose:
            print('--{} | shape={}'.format(layer_name, pool.get_shape()))
        return pool

    def build_model(self, input_img):
        if self.verbose:
            import re
            print('\nBUILDING VGG-19 NETWORK')
        net = {}
        _, h, w, d = input_img.shape

        if self.verbose: 
            print('Loading the model weights...')
        vgg_rawnet = scipy.io.loadmat(self.model_weights)
        vgg_layers = vgg_rawnet['layers'][0]

        if self.verbose: 
            print('Constructing the layers...')
        current_layer = net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

        for idx, layer in enumerate(self.VGG19_LAYERS):
            current_layer_name = layer[:4]
            if current_layer_name == 'conv':
                if self.verbose and re.match('conv\d_1', layer):
                    print("LAYER GROUP {}".format(layer[4]))
                current_layer = self._conv_layer(layer, current_layer, W=self.get_weights(vgg_layers, idx))
            elif current_layer_name == 'relu':
                current_layer = self._relu_layer(layer, current_layer, b=self.get_bias(vgg_layers, idx-1))
            elif current_layer_name == 'pool':
                current_layer = self._pool_layer(layer, current_layer)
            net[layer] = current_layer
        return net

    def get_weights(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)
        return W

    def get_bias(self, vgg_layers, i):
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return b