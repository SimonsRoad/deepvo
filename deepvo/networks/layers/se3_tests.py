"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: Ankur Handa
"""
import math
import unittest
import numpy as np
import tensorflow as tf

from se3 import *
from sophus import *
from pyquaternion import Quaternion


class SE3CompositeLayerTest(unittest.TestCase):
    def test_to_rotation_mtr_zero(self):
        x = tf.placeholder(tf.float32, shape=(None, 6))
        out = layer_matrix_rt(x)
        with tf.Session('') as sess:
            mtr = sess.run(out, feed_dict={x: np.zeros((1, 6))})
            np.testing.assert_array_equal(mtr, [[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]])

            mtr = sess.run(out, feed_dict={x: np.zeros((2, 6))})
            np.testing.assert_array_equal(mtr, [[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]] * 2)

    def test_to_rotation_mtr(self):
        x = tf.placeholder(tf.float32, shape=(None, 6))
        out_R = layer_matrix_rt(x)
        out_xyzq = layer_xyzq(out_R)
        with tf.Session('') as sess:
            # ---- TEST : Rotation X ----
            mtr_gt = [SE3.rotX(math.pi).matrix()]
            mtr = sess.run(out_R, feed_dict={x: [
                [math.pi, 0, 0, 0, 0, 0]
            ]})
            np.testing.assert_array_almost_equal(mtr, mtr_gt, decimal=4)

            # ---- TEST : Rotation Y + Quaternion ----
            mtr_gt = [SE3.rotY(math.pi).matrix()]
            mtr = sess.run(out_R, feed_dict={x: [
                [0, math.pi, 0, 0, 0, 0]
            ]})
            np.testing.assert_array_almost_equal(mtr, mtr_gt, decimal=4)

            mtr_gt = [SE3.rotY(math.pi / 2.0).matrix()]
            mtr, xyzq = sess.run([out_R, out_xyzq], feed_dict={x: [
                [0, math.pi / 2.0, 0, 0, 0, 0]
            ]})
            np.testing.assert_array_almost_equal(mtr, mtr_gt, decimal=4)
            np.testing.assert_array_almost_equal(mtr, [[[0, 0, 1, 0],
                                                        [0, 1, 0, 0],
                                                        [-1, 0, 0, 0],
                                                        [0, 0, 0, 1]]], decimal=4)
            # quaternion (0.707, 0, 0.707, 0)
            np.testing.assert_array_almost_equal(xyzq, [[0.707, 0, 0.707, 0, 0, 0, 0]], decimal=3)

            # ---- TEST : Rotation Z ----
            mtr_gt = [SE3.rotZ(math.pi).matrix()]
            mtr = sess.run(out_R, feed_dict={x: [
                [0, 0, math.pi, 0, 0, 0]
            ]})
            np.testing.assert_array_almost_equal(mtr, mtr_gt, decimal=4)

            # ---- TEST : Rotation XZ + Quaternion ----
            mtr, xyzq = sess.run([out_R, out_xyzq], feed_dict={x: [
                [1.5759287, 0, 1.0373201, 0, 0, 0]
            ]})
            np.testing.assert_array_almost_equal(mtr, [[[0.6037976, -0.5226062, 0.6019229, 0],
                                                       [0.5226062, -0.3106623, -0.7939595, 0],
                                                       [0.6019229, 0.7939595, 0.0855401, 0],
                                                       [0, 0, 0, 1]]], decimal=4)
            # quaternion [0.587085, 0.6761878, 0, 0.4450856]
            np.testing.assert_array_almost_equal(xyzq, [[0.587085, 0.6761878, 0, 0.4450856, 0, 0, 0]], decimal=3)

    def test_se3composite(self):
        x = tf.placeholder(tf.float32, shape=(1, 6))
        init_s = tf.placeholder(tf.float32, shape=(1, 4, 4))
        l_se3comp = SE3CompositeLayer()
        outputs, state = tf.nn.static_rnn(cell=l_se3comp,
                                          inputs=[x],
                                          initial_state=init_s,
                                          dtype=tf.float32)
        with tf.Session('') as sess:
            o, s = sess.run([outputs, state], feed_dict={
                x: [[0, 0, 0, 1, 0, 0]],
                init_s: [np.identity(4, np.float32)]
            })
            o, s = sess.run([outputs, state], feed_dict={
                x: [[0, 0, 0, 1, 0, 0]],
                init_s: s
            })
            np.testing.assert_array_almost_equal(o, [[[1, 0, 0, 0, 2, 0, 0]]])

            o, s = sess.run([outputs, state], feed_dict={
                x: [[0, 0, 0, 1, 1, 0]],
                init_s: s
            })
            np.testing.assert_array_almost_equal(o, [[[1, 0, 0, 0, 3, 1, 0]]])

            q = Quaternion(o[0][0][:4])
            y, p, r = q.yaw_pitch_roll
            pass
