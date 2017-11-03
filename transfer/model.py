import os
from subprocess import call

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input
from keras import layers
import pandas as pd
from tqdm import tqdm

from transfer.resnet50 import get_pre_post_model, get_pre_post_model_extra, get_final_model

def gen_minibatches(array_dir, batch_size):
    # reset seed for multiprocessing issues
    np.random.seed()

    array_files = sorted(os.listdir(array_dir))
    array_names = list(filter(lambda x: r'-img-' in x, array_files))
    label_names = list(filter(lambda x: r'-label-' in x, array_files))

    xy_names = list(zip(array_names, label_names))

    while True:
        # in place shuffle
        np.random.shuffle(xy_names)
        xy_names_mb = xy_names[:batch_size]

        arrays = []
        labels = []
        for array_name, label_name in xy_names_mb:
            arrays.append(np.load(os.path.join(array_dir, array_name)))
            labels.append(np.load(os.path.join(array_dir, label_name)))

        yield np.array(arrays), np.array(labels)


def train_model(project, extra_conv = False):
    if extra_conv:
        label = 'extra'
        weight_label = '-' + label + '-weights-'
    else:
        label = 'resnet'
        weight_label = '-' + label + '-weights-'
    weights_name = project['name'] + weight_label + str(project['model_round']) +'.hdf5'
    source_path = project['path']

    if extra_conv == False:
        project['model_round'] += 1

    img_classes = [d for d in os.listdir(project['img_path']) if os.path.isdir(os.path.join(project['img_path'], d))]

    weights_path = os.path.join(source_path, 'weights')
    pre_model_path_test = os.path.join(source_path, 'pre_model', 'test')
    pre_model_path_train = os.path.join(source_path, 'pre_model', 'train')
    call(['mkdir', '-p', weights_path])

    number_train_samples = len(os.listdir(pre_model_path_train))
    number_test_samples = len(os.listdir(pre_model_path_test))

    gen_train = gen_minibatches(pre_model_path_train, project['batch_size'])
    gen_test = gen_minibatches(pre_model_path_test, project['batch_size'])

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']

    if extra_conv:
        pre_post_function = get_pre_post_model_extra
    else:
        pre_post_function = get_pre_post_model

    pre_model, model = pre_post_function(img_dim,
                                         conv_dim,
                                         len(img_classes),
                                         model_weights = project['resnet_best_weights'])

    optimizer = Adam(lr = project[label + '_learning_rate'])
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['categorical_accuracy'])

    weights_checkpoint_file = weights_name.split('.')[0] + "-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(weights_path, weights_checkpoint_file),
                                 monitor='val_categorical_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    callbacks_list = [checkpoint]
    steps_per_epoch = (number_train_samples // project['batch_size'])
    validation_steps = (number_test_samples // project['batch_size'])

    model.fit_generator(gen_train,
                        steps_per_epoch = steps_per_epoch,
                        epochs = project['epochs'],
                        verbose = 2,
                        validation_data = gen_test,
                        validation_steps = validation_steps,
                        initial_epoch = 0,
                        callbacks = callbacks_list)

    model.save_weights(os.path.join(weights_path, weights_name))
    weights_names = os.listdir(weights_path)
    max_val = -1
    max_i = -1
    for i, name in enumerate(weights_names):
        if name.find(weights_name.split('.')[0]) >= 0:
            if (name.find(weight_label) >= 0) and (name.find('improvement') >= 0):
                val = int(name.split('.')[1])
                if val > max_val:
                    max_val = val
                    max_i = i

    if max_i == -1:
        best_weights = os.path.join(weights_path, weights_name)
    else:
        best_weights = os.path.join(weights_path, weights_names[max_i])

    project['number_categories'] = len(img_classes)
    project[label + '_learning_rate'] = project[label + '_learning_rate'] / project[label + '_learning_rate_modifier']
    project[label + '_best_weights'] = best_weights
    project[label + '_last_weights'] = os.path.join(weights_path, weights_name)

    return project
