#!/usr/bin/env python3
create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):

     for layer_size, activation in zip(layer_sizes, activations):
        pred=create_layer(x, layer_size, activation)
    return pred
