import os
from subprocess import call

import numpy as np
from keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

from transfer.resnet50 import get_pre_model

def gen_array_from_dir(array_dir):
    array_files = sorted(os.listdir(array_dir))

    array_names = list(filter(lambda x: r'-img-' in x, array_files))
    label_names = list(filter(lambda x: r'-label-' in x, array_files))

    assert len(array_names) == len(label_names)

    for arr_name, lab_name in zip(array_names, label_names):
        X = np.load(os.path.join(array_dir, arr_name))
        Y = np.load(os.path.join(array_dir, lab_name))
        yield X, Y, arr_name, lab_name


def pre_model(project):

    img_dim = 224 * project['img_size']
    print('Predicting pre-model for test')
    val_pre_model(project['path'], 'array', 'test', img_dim)
    print('Predicting pre-model for train')
    val_pre_model(project['path'], 'augmented', 'train', img_dim)

    project['is_pre_model'] = True
    return project

def val_pre_model(source_path, folder, val_group, img_dim):

    array_path = os.path.join(source_path, folder, val_group)
    pre_model_path = os.path.join(source_path, 'pre_model', val_group)
    call(['rm', '-rf', pre_model_path])
    call(['mkdir', '-p', pre_model_path])

    popped, pre_model = get_pre_model(img_dim)

    for (array, label, array_name, label_name) in tqdm(gen_array_from_dir(array_path)):
        array = preprocess_input(array[np.newaxis].astype(np.float32))
        array_pre_model = np.squeeze(pre_model.predict(array, batch_size=1))

        array_name = array_name.split('.')[0]
        label_name = label_name.split('.')[0]

        img_pre_model_path = os.path.join(pre_model_path, array_name)
        label_pre_model_path = os.path.join(pre_model_path, label_name)

        np.save(img_pre_model_path, array_pre_model)
        np.save(label_pre_model_path, label)
