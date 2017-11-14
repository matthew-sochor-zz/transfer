import os
from subprocess import call

import numpy as np
from keras.layers import Input, Activation, Conv2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
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


def get_pre_model(img_dim):
    resnet = get_resnet_model(img_dim)
    popped, pre_model = pop_layer(resnet, 12)
    return popped, pre_model


def get_pre_post_model(img_dim, conv_dim, number_categories, model_weights = None, extra_conv = False, end_copy = False):

    popped, pre_model = get_pre_model(img_dim)

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

    if extra_conv:
        x = Conv2D(512, (1, 1), name = 'conv_heatmap')(x_in_2)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = AveragePooling2D((7, 7), name = 'avg_pool')(x)
    else:
        x = AveragePooling2D((7, 7), name = 'avg_pool')(x_in_2)

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(number_categories, activation = 'softmax')(x)

    end_model = Model(x_in_2, x)

    x_in_3 = Input(shape = input_dims)
    x = mid_model(x_in_3)
    x = end_model(x)
    post_model = Model(x_in_3, x)

    if model_weights is not None:
        print('Loading model weights:', model_weights)
        post_model.load_weights(model_weights)

    if end_copy:
        x_in_2_copy = Input(shape = input_dims)

        if extra_conv:
            x = Conv2D(512, (1, 1), name = 'conv_heatmap')(x_in_2_copy)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = AveragePooling2D((7, 7), name = 'avg_pool')(x)
        else:
            x = AveragePooling2D((7, 7), name = 'avg_pool')(x_in_2_copy)

        x = GlobalAveragePooling2D()(x)

        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(number_categories, activation = 'softmax')(x)

        end_model_copy = Model(x_in_2_copy, x)
        for i in range(len(post_model.layers[2].layers)):
            end_model_copy.layers[i].set_weights(end_model.layers[i].get_weights())
        return pre_model, post_model, end_model_copy
    else:
        return pre_model, post_model


def get_final_model(img_dim, conv_dim, number_categories, weights, extra_conv = False):

    pre_model, post_model = get_pre_post_model(img_dim, conv_dim, number_categories, model_weights = weights, extra_conv = extra_conv)
    x_in = Input(shape = (img_dim, img_dim, 3))
    x = pre_model(x_in)
    x = post_model(x)
    final_model = Model(x_in, x)
    return final_model


def get_final_model_separated(img_dim, conv_dim, number_categories, weights, extra_conv = True):
    pre_model, post_model, end_model = get_pre_post_model(img_dim, conv_dim, number_categories, model_weights = weights, extra_conv = extra_conv, end_copy = True)

    x_in = Input(shape = (img_dim, img_dim, 3))
    x = pre_model(x_in)
    x = post_model.layers[1](x)
    pre_mid_model = Model(x_in, x)

    return pre_mid_model, end_model


def get_pre_post_model_extra(img_dim, conv_dim, number_categories, model_weights = None):
    pre_model, post_model = get_pre_post_model(img_dim,
                                               conv_dim,
                                               number_categories,
                                               model_weights)

    pre_model_extra, post_model_extra = get_pre_post_model(img_dim,
                                                           conv_dim,
                                                           number_categories,
                                                           None,
                                                           True)

    for i, layer in enumerate(post_model.layers[1:]):
        for ii, inner_layer in enumerate(layer.layers):
            for j,layer_extra in enumerate(post_model_extra.layers[1:]):
                for jj, inner_layer_extra in enumerate(layer_extra.layers):
                    if inner_layer.name == inner_layer_extra.name:
                        post_model_extra.layers[j + 1].layers[jj].set_weights(post_model.layers[i + 1].layers[ii].get_weights())

    for i, layer in enumerate(post_model.layers[1].layers):
        post_model.layers[1].layers[i].trainable = False

    return pre_model_extra, post_model_extra


def export_model(project):
    model_name = project['name'] + '-' + str(project['model_round']) +'.hdf5'
    last_model_name = 'last-' + model_name
    best_model_name = 'best-' + model_name
    model_path = os.path.join(project['path'], 'model')
    call(['mkdir', '-p', model_path])

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']

    last_model = get_final_model(img_dim, conv_dim, project['number_categories'], project['last_weights'])
    best_model = get_final_model(img_dim, conv_dim, project['number_categories'], project['best_weights'])

    last_model.save(os.path.join(model_path, last_model_name))
    best_model.save(os.path.join(model_path, best_model_name))
