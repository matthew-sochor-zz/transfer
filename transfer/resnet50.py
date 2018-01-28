import os

import numpy as np
from keras.layers import Input, Activation, Conv2D, AveragePooling2D, Flatten, BatchNormalization, Dropout, Dense
from keras.models import Model
from keras import layers
from keras.applications.resnet50 import ResNet50


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


def get_resnet_model(img_dim):
    array_input = Input(shape=(img_dim, img_dim, 3))
    resnet = ResNet50(include_top=False,
                     weights='imagenet',
                     input_tensor=array_input,
                     pooling='avg')
    return resnet


def get_resnet_pre_model(img_dim):
    resnet = get_resnet_model(img_dim)
    popped, pre_model = pop_layer(resnet, 12)
    return popped, pre_model


def get_resnet_pre_post_model(img_dim, conv_dim, number_categories, model_weights = None):

    popped, pre_model = get_resnet_pre_model(img_dim)

    input_dims = (conv_dim, conv_dim, 2048)
    # Take last 12 layers from resnet 50 with their starting weights!
    x_in = Input(shape = input_dims)

    x = popped[11](x_in)
    x = popped[10](x)
    x = Activation('relu')(x)

    x = popped[8](x)
    x = popped[7](x)
    x = Activation('relu')(x)

    x = popped[5](x)
    x = popped[4](x)

    x = layers.add([x, x_in])
    x = Activation('relu')(x)
    mid_model = Model(x_in, x)

    x_in_2 = Input(shape = input_dims)

    x = AveragePooling2D((7, 7), name = 'avg_pool')(x_in_2)
    x = Flatten()(x)
    x = Dense(number_categories, activation = 'softmax')(x)

    end_model = Model(x_in_2, x)

    x_in_3 = Input(shape = input_dims)
    x = mid_model(x_in_3)
    x = end_model(x)
    post_model = Model(x_in_3, x)

    if model_weights is not None:
        print('Loading model weights:', model_weights)
        post_model.load_weights(model_weights)

    return pre_model, post_model


def get_resnet_final_model(img_dim, conv_dim, number_categories, weights, is_final):

    if is_final:
        pre_post_weights = None
    else:
        pre_post_weights = weights
    pre_model, post_model = get_resnet_pre_post_model(img_dim, conv_dim, number_categories, model_weights = pre_post_weights)
    x_in = Input(shape = (img_dim, img_dim, 3))
    x = pre_model(x_in)
    x = post_model(x)
    final_model = Model(x_in, x)
    if is_final:
        print('Loading model weights:', weights)
        final_model.load_weights(weights)
    return final_model
