import os
from subprocess import call

import numpy as np
from keras.layers import Input, Activation, concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras import layers
from keras.applications.inception_v3 import InceptionV3


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


def get_inception_v3_model(img_dim):
    array_input = Input(shape=(img_dim, img_dim, 3))
    inception_v3 = InceptionV3(include_top=True,
                        weights='imagenet',
                        input_tensor=array_input)
    return inception_v3


def get_inception_v3_pre_model(img_dim):
    inception_v3 = get_inception_v3_model(img_dim)
    popped, pre_model = pop_layer(inception_v3, 33)
    return popped, pre_model


def get_inception_v3_pre_post_model(img_dim, conv_dim, number_categories, model_weights = None):

    popped, pre_model = get_inception_v3_pre_model(img_dim)

    input_dims = (conv_dim, conv_dim, 2048)
    # Take last 33 layers from inception_v3 with their starting weights!
    mixed_9 = Input(shape = input_dims)

    branch1x1 = popped[18](mixed_9)
    branch1x1 = popped[12](branch1x1)
    branch1x1 = popped[6](branch1x1)

    branch3x3 = popped[29](mixed_9)
    branch3x3 = popped[27](branch3x3)
    branch3x3 = popped[25](branch3x3)

    branch3x3_1 = popped[23](branch3x3)
    branch3x3_1 = popped[17](branch3x3_1)
    branch3x3_1 = popped[11](branch3x3_1)

    branch3x3_2 = popped[22](branch3x3)
    branch3x3_2 = popped[16](branch3x3_2)
    branch3x3_2 = popped[10](branch3x3_2)

    branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_1')

    branch3x3dbl = popped[32](mixed_9)
    branch3x3dbl = popped[31](branch3x3dbl)
    branch3x3dbl = popped[30](branch3x3dbl)
    branch3x3dbl = popped[28](branch3x3dbl)
    branch3x3dbl = popped[26](branch3x3dbl)
    branch3x3dbl = popped[24](branch3x3dbl)

    branch3x3dbl_1 = popped[21](branch3x3dbl)
    branch3x3dbl_1 = popped[15](branch3x3dbl_1)
    branch3x3dbl_1 = popped[9](branch3x3dbl_1)

    branch3x3dbl_2 = popped[20](branch3x3dbl)
    branch3x3dbl_2 = popped[14](branch3x3dbl_2)
    branch3x3dbl_2 = popped[8](branch3x3dbl_2)

    branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3, name='concatenate_4')

    branch_pool = popped[19](mixed_9)
    branch_pool = popped[13](branch_pool)
    branch_pool = popped[7](branch_pool)
    branch_pool = popped[3](branch_pool)

    x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed10')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(number_categories, activation='softmax', name='predictions')(x)

    post_model = Model(mixed_9, x)
    if model_weights is not None:
        print('Loading model weights:', model_weights)
        post_model.load_weights(model_weights)

    return pre_model, post_model


def get_inception_v3_final_model(img_dim, conv_dim, number_categories, weights, is_final):

    if is_final:
        pre_post_weights = None
    else:
        pre_post_weights = weights
    pre_model, post_model = get_inception_v3_pre_post_model(img_dim, conv_dim, number_categories, model_weights = pre_post_weights)
    x_in = Input(shape = (img_dim, img_dim, 3))
    x = pre_model(x_in)
    x = post_model(x)
    final_model = Model(x_in, x)
    if is_final:
        print('Loading model weights:', weights)
        final_model.load_weights(weights)
    return final_model
