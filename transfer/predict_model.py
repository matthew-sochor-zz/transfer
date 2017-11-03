import os

import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input
import pandas as pd
from tqdm import tqdm

from transfer.model import get_final_model

def prep_from_image(file_name, img_dim):
    img = np.array(load_img(file_name, target_size = (img_dim, img_dim, 3)))
    return preprocess_input(img[np.newaxis].astype(np.float32))


def gen_from_directory(directory, img_dim):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

    for file_name in file_names:
        if ((file_name.find('.jpg') > 0) or (file_name.find('.jpeg') > 0) or (file_name.find('.png') > 0)):
            yield prep_from_image(os.path.join(directory, file_name), img_dim), os.path.join(directory, file_name)


def predict_model(project, weights, user_files, model = None):

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    if model is None:
        model = get_final_model(img_dim, conv_dim, project['number_categories'], project[weights])

    img_classes = [d for d in os.listdir(project['img_path']) if os.path.isdir(os.path.join(project['img_path'],d))]

    output = []
    user_files = os.path.expanduser(user_files)
    if os.path.isdir(user_files):
        for img, file_name in tqdm(gen_from_directory(user_files, img_dim)):
            predicted = model.predict(img)
            pred_list = list(predicted[0])
            output.append([file_name, img_classes[np.argmax(predicted)]] + pred_list)

    elif ((user_files.find('.jpg') > 0) or (user_files.find('.jpeg') > 0) or (user_files.find('.png') > 0)):
        img = prep_from_image(user_files, img_dim)
        predicted = model.predict(img)
        pred_list = list(predicted[0])
        output.append([user_files, img_classes[np.argmax(predicted)]] + pred_list)

    else:
        print('Should either be a directory or a .jpg, .jpeg, and .png')
        return


    if len(output) > 0:
        columns = ['file_name', 'predicted'] + img_classes
        pred_df = pd.DataFrame(output, columns = columns)

        predictions_file = os.path.join(project['path'], project['name'] + '_' + weights + '_predictions.csv')
        if os.path.isfile(predictions_file):
            old_pred_df = pd.read_csv(predictions_file)
            pred_df = pd.concat([pred_df, old_pred_df])

        pred_df.to_csv(predictions_file, index = False)
        print('Predictions saved to:', predictions_file)

    else:
        print('No image files found.')
