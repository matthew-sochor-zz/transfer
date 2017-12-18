import os
from subprocess import call

import numpy as np
from keras.layers import Input, Activation, Conv2D, AveragePooling2D, Flatten, BatchNormalization, Dropout, Dense
from keras.models import Model
from keras import layers
from keras.applications.xception import Xception


def pop_layer(model, count=1):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    popped = [model.layers.pop() for i in range(count)]

    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]

    model.container_nodes = sorted([l.name for l in model.layers])
    model.built = True

    return popped, model


def get_xception_model(img_dim):
    array_input = Input(shape=(img_dim, img_dim, 3))
    xception = Xception(include_top=True,
                     weights='imagenet',
                     input_tensor=array_input,
                     pooling='avg')
    return xception


def get_xception_pre_model(img_dim):
    xception = get_xception_model(img_dim)
    popped, pre_model = pop_layer(xception, 8)
    return popped, pre_model


def get_xception_pre_post_model(img_dim, conv_dim, number_categories, model_weights = None):

    popped, pre_model = get_xception_pre_model(img_dim)

    input_dims = (conv_dim, conv_dim, 1024)
    # Take last 8 layers from xception with their starting weights!
    x_in = Input(shape = input_dims)
    x = popped[7](x_in)
    x = popped[6](x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = popped[4](x)
    x = popped[3](x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = popped[1](x)
    x = Dense(number_categories, activation = 'softmax', name='predictions')(x)

    post_model = Model(x_in, x)

    if model_weights is not None:
        print('Loading model weights:', model_weights)
        post_model.load_weights(model_weights)

    return pre_model, post_model


def get_xception_final_model(img_dim, conv_dim, number_categories, weights, is_final):

    if is_final:
        pre_post_weights = None
    else:
        pre_post_weights = weights
    pre_model, post_model = get_xception_pre_post_model(img_dim, conv_dim, number_categories, model_weights = pre_post_weights)
    x_in = Input(shape = (img_dim, img_dim, 3))
    x = pre_model(x_in)
    x = post_model(x)
    final_model = Model(x_in, x)
    if is_final:
        print('Loading model weights:', weights)
        final_model.load_weights(weights)
    return final_model
