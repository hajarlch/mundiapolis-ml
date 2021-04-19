#!/usr/bin/env python3
import tensorflow as tf

def calculate_accuracy(y, y_pred):

    pred = tf.equal(tf.argmax(y_pred, 1),tf.argmax(y, 1))
    m=tf.reduce_mean(tf.cast(prediction, tf.float32))

    return m