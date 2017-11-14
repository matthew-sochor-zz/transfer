import os
from subprocess import call

import numpy as np
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageFilter
from colorama import init
from termcolor import colored

from transfer.resnet50 import get_final_model, get_final_model_separated


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(input_model, image, mid_image, category_index, layer_name, nb_classes):
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))

    conv_output = model.layers[0].layers[1].output
    loss = K.sum(model.layers[-1].output)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([mid_image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    img_size = np.shape(image)[1]
    cam = cv2.resize(cam, (img_size, img_size))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def prep_from_image(file_name, img_dim):
    img = np.array(load_img(file_name, target_size = (img_dim, img_dim, 3)))
    return preprocess_input(img[np.newaxis].astype(np.float32))


def gen_from_directory(directory, img_dim):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

    for file_name in file_names:
        if ((file_name.find('.jpg') > 0) or (file_name.find('.jpeg') > 0) or (file_name.find('.png') > 0)):
            yield prep_from_image(os.path.join(directory, file_name), img_dim), os.path.join(directory, file_name)


def predict_model(project, weights, user_files, model = None, extra_conv = False):

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    if model is None:
        model = get_final_model(img_dim, conv_dim, project['number_categories'], project[weights], extra_conv = extra_conv)

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
        print(colored('Should either be a directory or a .jpg, .jpeg, and .png', 'red'))
        return


    if len(output) > 0:
        columns = ['file_name', 'predicted'] + img_classes
        pred_df = pd.DataFrame(output, columns = columns)

        predictions_file = os.path.join(project['path'], project['name'] + '_' + weights + '_predictions.csv')
        if os.path.isfile(predictions_file):
            old_pred_df = pd.read_csv(predictions_file)
            pred_df = pd.concat([pred_df, old_pred_df])

        pred_df.to_csv(predictions_file, index = False)
        print('Predictions saved to:', colored(predictions_file, 'cyan'))

    else:
        print(colored('No image files found.', 'red'))


def predict_heatmap(pre_mid_model, end_model, file_name, img, img_classes, heatmap_path, low_val = 0.5):
    pre_mid_predicted = pre_mid_model.predict(img)
    predicted = end_model.predict(pre_mid_predicted)
    pred_list = list(predicted[0])
    predicted_class_index = np.argmax(predicted)
    cam, heatmap = grad_cam(end_model, img, pre_mid_predicted, predicted_class_index, 'conv_heatmap', len(img_classes))
    file_name_pre = os.path.split(file_name)[1].split('.')[0]
    heatmap_file_name = os.path.join(heatmap_path, img_classes[predicted_class_index] + '_' + file_name_pre + '.png')
    heatmap_overlay_file_name = os.path.join(heatmap_path, img_classes[predicted_class_index] + '_overlay_' + file_name_pre + '.png')

    heatmap_orig = heatmap.copy()
    heatmap[heatmap < (np.mean(heatmap))] = low_val
    heatmap[heatmap > low_val] = 1

    heatmap_im = Image.fromarray(np.uint8(heatmap))

    edges = np.array(heatmap_im.filter(ImageFilter.FIND_EDGES))
    edges[edges > 0] = 255

    for s in [-1, 1]:
        for a in [0, 1]:
            eroll = np.roll(edges, shift=1, axis=1)
            edges[eroll > 0] = 255
    nimg = np.array(img[0])

    b = np.squeeze(nimg[:,:, 0]) * heatmap
    b[edges == 255] = 255
    m = np.squeeze(nimg[:,:, 1]) * heatmap
    m[edges == 255] = 0
    r = np.squeeze(nimg[:,:, 2]) * heatmap
    r[edges == 255] = 0

    mimg = np.zeros(np.shape(img[0]))

    mimg[:,:,0] = b
    mimg[:,:,1] = m
    mimg[:,:,2] = r

    #mimg = nimg.astype(np.uint8)
    cv2.imwrite(heatmap_file_name, mimg)
    cv2.imwrite(heatmap_overlay_file_name, cam)
    return [file_name, img_classes[predicted_class_index]] + pred_list


def predict_activation_model(project, weights, user_files, model = None):

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    if model is None:
        pre_mid_model, end_model = get_final_model_separated(img_dim,
                                                            conv_dim,
                                                            project['number_categories'],
                                                            project[weights],
                                                            extra_conv = True)

    img_classes = [d for d in os.listdir(project['img_path']) if os.path.isdir(os.path.join(project['img_path'],d))]

    heatmap_path = os.path.join(project['path'], 'heatmaps')
    call(['mkdir', '-p', heatmap_path])
    output = []
    user_files = os.path.expanduser(user_files)
    if os.path.isdir(user_files):
        for img, file_name in tqdm(gen_from_directory(user_files, img_dim)):
            out = predict_heatmap(pre_mid_model, end_model, file_name, img, img_classes, heatmap_path)
            output.append(out)

    elif ((user_files.find('.jpg') > 0) or (user_files.find('.jpeg') > 0) or (user_files.find('.png') > 0)):
        img = prep_from_image(user_files, img_dim)
        out = predict_heatmap(pre_mid_model, end_model, user_files, img, img_classes, heatmap_path)
        output.append(out)

    else:
        print(colored('Should either be a directory or a .jpg, .jpeg, and .png', 'red'))
        return


    if len(output) > 0:
        columns = ['file_name', 'predicted'] + img_classes
        pred_df = pd.DataFrame(output, columns = columns)

        predictions_file = os.path.join(project['path'], project['name'] + '_' + weights + '_predictions.csv')
        if os.path.isfile(predictions_file):
            old_pred_df = pd.read_csv(predictions_file)
            pred_df = pd.concat([pred_df, old_pred_df])

        pred_df.to_csv(predictions_file, index = False)
        print('Predictions saved to:', colored(predictions_file, 'cyan'))

    else:
        print(colored('No image files found.', 'red'))
