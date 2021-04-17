#!/usr/bin/env python3
import tensorflow as tf
def create_layer(prev, n, activation):
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=init,
        activation=activation,
        name='Layer'
    )
    ly = layer(prev)
    return ly