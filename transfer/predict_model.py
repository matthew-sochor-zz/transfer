import os
from subprocess import call

import numpy as np
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from colorama import init
from termcolor import colored

from transfer.resnet50 import get_resnet_final_model
from transfer.xception import get_xception_final_model
from transfer.inception_v3 import get_inception_v3_final_model
from transfer.augment_arrays import gen_augment_arrays


def prep_from_image(file_name, img_dim, augmentations):
    img = np.array(load_img(file_name, target_size = (img_dim, img_dim, 3)))

    return gen_augment_arrays(img, np.array([]), augmentations)


def gen_from_directory(directory, img_dim, project):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

    for file_name in file_names:
        if ((file_name.find('.jpg') > 0) or (file_name.find('.jpeg') > 0) or (file_name.find('.png') > 0)):
            yield prep_from_image(os.path.join(directory, file_name), img_dim, project['augmentations']), os.path.join(directory, file_name)


def multi_predict(aug_gen, models, architecture):
    predicted = []
    for img, _ in aug_gen:
        if architecture == 'resnet50':
            img = resnet_preprocess_input(img[np.newaxis].astype(np.float32))
        elif architecture == 'xception':
            img = xception_preprocess_input(img[np.newaxis].astype(np.float32))
        else:
            img = inception_v3_preprocess_input(img[np.newaxis].astype(np.float32))
        for model in models:
            predicted.append(model.predict(img))
    predicted = np.array(predicted).sum(axis=0)
    pred_list = list(predicted[0])
    return predicted, pred_list

def predict_model(project, weights, user_files):

    img_dim = project['img_dim'] * project['img_size']
    conv_dim = project['conv_dim'] * project['img_size']
    models = []
    for weight in project[weights]:
        if project['architecture'] == 'resnet50':
            models.append(get_resnet_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))
        elif project['architecture'] == 'xception':
            models.append(get_xception_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))
        else:
            models.append(get_inception_v3_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))

    output = []
    user_files = os.path.expanduser(user_files)
    if os.path.isdir(user_files):
        for aug_gen, file_name in tqdm(gen_from_directory(user_files, img_dim, project)):
            predicted, pred_list = multi_predict(aug_gen, models, project['architecture'])
            output.append([project[weights], file_name, project['categories'][np.argmax(predicted)]] + pred_list)

    elif ((user_files.find('.jpg') > 0) or (user_files.find('.jpeg') > 0) or (user_files.find('.png') > 0)):
        aug_gen = prep_from_image(user_files, img_dim, project['augmentations'])
        predicted, pred_list = multi_predict(aug_gen, models, project['architecture'])
        output.append([project[weights], user_files, project['categories'][np.argmax(predicted)]] + pred_list)

    else:
        print(colored('Should either be a directory or a .jpg, .jpeg, and .png', 'red'))
        return


    if len(output) > 0:
        columns = ['weights_used','file_name', 'predicted'] + project['categories']
        pred_df = pd.DataFrame(output, columns = columns)

        predictions_file = os.path.join(project['path'], project['name'] + '_' + weights + '_predictions.csv')
        if os.path.isfile(predictions_file):
            old_pred_df = pd.read_csv(predictions_file)
            pred_df = pd.concat([pred_df, old_pred_df])

        pred_df.to_csv(predictions_file, index = False)
        print('Predictions saved to:', colored(predictions_file, 'cyan'))

    else:
        print(colored('No image files found.', 'red'))
