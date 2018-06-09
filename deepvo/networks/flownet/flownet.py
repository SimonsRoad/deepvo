import abc
import os
import numpy as np
import tensorflow as tf


class Flownet:
    def __init__(self):
        pass

    @abc.abstractmethod
    def build(self, input_ab):
        pass

    def build_ab(self, input_a, input_b):
        x = tf.concat([input_a, input_b], 3, name='inp_concat')
        return self.build(x)

    @abc.abstractmethod
    def get_weight_path(self):
        pass

    def initailize(self, tf_sess):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', self.get_weight_path())
        with open(path, 'rb') as f:
            weights = np.load(f)

        ops = []
        graph = tf.get_default_graph()
        for v_name, v_weights in weights.iteritems():
            v_tensor = graph.get_tensor_by_name(FlownetS.NAME + '/' + v_name)
            v_assign_op = tf.assign(v_tensor, v_weights)
            ops.append(v_assign_op)
        tf_sess.run(ops)


class FlownetS(Flownet):
    NAME = 'flownet_s'

    def __init__(self):
        Flownet.__init__(self)

    def build(self, input_ab):
        h, w = input_ab.shape[1:3]

        with tf.variable_scope(FlownetS.NAME, reuse=tf.AUTO_REUSE):
            x = tf.divide(input_ab, 255.0, name='inp_norm')
            assert x.shape.as_list()[1:] == [h, w, 6], x.shape

            conv_list = [
                ('conv', {'filters': 64, 'kernel_size': 7, 'strides': 2, 'padding': 'SAME', 'name': 'conv1'}),
                ('assert_shape', [h // 2, w // 2, 64]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv1/leaky_relu'}),

                ('conv', {'filters': 128, 'kernel_size': 5, 'strides': 2, 'padding': 'SAME', 'name': 'conv2'}),
                ('assert_shape', [h // 4, w // 4, 128]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv2/leaky_relu'}),

                ('conv', {'filters': 256, 'kernel_size': 5, 'strides': 2, 'padding': 'SAME', 'name': 'conv3'}),
                ('assert_shape', [h // 8, w // 8, 256]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv3/leaky_relu'}),

                ('conv', {'filters': 256, 'kernel_size': 3, 'strides': 1, 'padding': 'SAME', 'name': 'conv3_1'}),
                ('assert_shape', [h // 8, w // 8, 256]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv3_1/leaky_relu'}),

                ('conv', {'filters': 512, 'kernel_size': 3, 'strides': 2, 'padding': 'SAME', 'name': 'conv4'}),
                ('assert_shape', [h // 16, w // 16, 512]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv4/leaky_relu'}),

                ('conv', {'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'SAME', 'name': 'conv4_1'}),
                ('assert_shape', [h // 16, w // 16, 512]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv4_1/leaky_relu'}),

                ('conv', {'filters': 512, 'kernel_size': 3, 'strides': 2, 'padding': 'SAME', 'name': 'conv5'}),
                ('assert_shape', [h // 32, w // 32, 512]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv5/leaky_relu'}),

                ('conv', {'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'SAME', 'name': 'conv5_1'}),
                ('assert_shape', [h // 32, w // 32, 512]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv5_1/leaky_relu'}),

                ('conv', {'filters': 1024, 'kernel_size': 3, 'strides': 2, 'padding': 'SAME', 'name': 'conv6'}),
                ('assert_shape', [h // 64, w // 64, 1024]),
                ('leaky_relu', {'alpha': 0.1, 'name': 'conv6/leaky_relu'})
            ]

            layers = {}
            for layer_type, args in conv_list:
                if layer_type == 'conv':
                    x = tf.layers.conv2d(x, **args)
                    layers[args['name']] = x
                elif layer_type == 'leaky_relu':
                    x = tf.nn.leaky_relu(x, **args)
                    layers[args['name']] = x
                elif layer_type == 'assert_shape':
                    assert x.shape.as_list()[1:] == args, x.shape
        return layers

    def get_weight_path(self):
        return 'models/flownet_v1_s/weight.npy'


class FlownetC(Flownet):
    def __init__(self):
        Flownet.__init__(self)


if __name__ == '__main__':
    img_a = tf.placeholder(tf.float32, shape=(None, 384, 512, 3))
    img_b = tf.placeholder(tf.float32, shape=(None, 384, 512, 3))

    flownet_s = FlownetS()
    flownet_s.build(img_a, img_b)

    tf_sess = tf.Session()
    flownet_s.initailize(tf_sess)
