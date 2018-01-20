import os
import shutil
from subprocess import call

import numpy as np
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tqdm import tqdm

from transfer.resnet50 import get_resnet_pre_model
from transfer.xception import get_xception_pre_model
from transfer.inception_v3 import get_inception_v3_pre_model

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

    img_dim = project['img_dim'] * project['img_size']
    print('Predicting pre-model')
    val_pre_model(project['path'], 'augmented', img_dim, project['architecture'])

    project['is_pre_model'] = True
    return project


def val_pre_model(source_path, folder, img_dim, architechture):

    array_path = os.path.join(source_path, folder)
    pre_model_path = os.path.join(source_path, 'pre_model')
#    call(['rm', '-rf', pre_model_path])
    shutil.rmtree(pre_model_path,ignore_errors=True)
#    call(['mkdir', '-p', pre_model_path])
    os.makedirs(pre_model_path)

    if architechture == 'resnet50':
        popped, pre_model = get_resnet_pre_model(img_dim)
    elif architechture == 'xception':
        popped, pre_model = get_xception_pre_model(img_dim)
    else:
        popped, pre_model = get_inception_v3_pre_model(img_dim)

    for (array, label, array_name, label_name) in tqdm(gen_array_from_dir(array_path)):
        if architechture == 'resnet50':
            array = resnet_preprocess_input(array[np.newaxis].astype(np.float32))
        elif architechture == 'xception':
            array = xception_preprocess_input(array[np.newaxis].astype(np.float32))
        else:
            array = inception_v3_preprocess_input(array[np.newaxis].astype(np.float32))
        array_pre_model = np.squeeze(pre_model.predict(array, batch_size=1))

        array_name = array_name.split('.')[0]
        label_name = label_name.split('.')[0]

        img_pre_model_path = os.path.join(pre_model_path, array_name)
        label_pre_model_path = os.path.join(pre_model_path, label_name)

        np.save(img_pre_model_path, array_pre_model)
        np.save(label_pre_model_path, label)
