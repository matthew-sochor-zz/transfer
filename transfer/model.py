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

from transfer.resnet50 import get_pre_post_model, get_final_model

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


def prep_from_image(file_name):
    img = np.array(load_img(file_name, target_size = (224, 224, 3)))
    return preprocess_input(img[np.newaxis].astype(np.float32))


def gen_from_directory(directory):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]
    
    for file_name in file_names:
        if ((file_name.find('.jpg') > 0) or (file_name.find('.jpeg') > 0) or (file_name.find('.png') > 0)):
            yield prep_from_image(os.path.join(directory, file_name)), os.path.join(directory, file_name)


def predict_model(project, weights, user_files, model = None):

    if model is None:
        img_dim = 224 * project['img_size']
        conv_dim = 7 * project['img_size']
        model = get_final_model(img_dim, conv_dim, project['number_categories'], project[weights])

    img_classes = [d for d in os.listdir(project['path']) if os.path.isdir(os.path.join(project['path'],d))]

    predictions = []
    file_names = []
    user_files = os.path.expanduser(user_files)
    if os.path.isdir(user_files):
        for img, file_name in tqdm(gen_from_directory(user_files)):
            predicted = model.predict(img)
            predictions.append(img_classes[np.argmax(predicted)])
            file_names.append(file_name)
        
    elif ((user_files.find('.jpg') > 0) or (user_files.find('.jpeg') > 0) or (user_files.find('.png') > 0)):
        img = prep_from_image(user_files)
        predicted = model.predict(img)
        predictions.append(img_classes[np.argmax(predicted)])
        file_names.append(user_files)

    else:
        print('Should either be a directory or a .jpg, .jpeg, and .png')
        return

    if len(predictions) > 0:

        pred_df = pd.DataFrame({'predictions': predictions, 'file_names': file_names})

        predictions_file = os.path.join(project['path'], '..', project['name'] + '_' + weights + '_predictions.csv')
        if os.path.isfile(predictions_file):
            old_pred_df = pd.read_csv(predictions_file)
            pred_df = pd.concat([pred_df, old_pred_df])
        
        pred_df.to_csv(predictions_file, index = False)
        print('Predictions saved to:', predictions_file)

    else:
        print('No image files found.')


def train_model(project):
    project['model_round'] += 1
    weights_name = project['name'] + '-weights-' + str(project['model_round']) +'.hdf5'
    source_path = project['path']

    img_classes = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path,d))]

    weights_path = os.path.join(source_path, '..', 'weights')
    pre_model_path_test = os.path.join(source_path, '..', 'pre_model', 'test')
    pre_model_path_train = os.path.join(source_path, '..', 'pre_model', 'train')
    call(['mkdir', '-p', weights_path])

    number_train_samples = len(os.listdir(pre_model_path_train))
    number_test_samples = len(os.listdir(pre_model_path_test))

    gen_train = gen_minibatches(pre_model_path_train, project['batch_size'])
    gen_test = gen_minibatches(pre_model_path_test, project['batch_size'])

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    pre_model, model = get_pre_post_model(img_dim, conv_dim, len(img_classes), model_weights = project['best_weights'])

    optimizer = Adam(lr = project['learning_rate'])
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['categorical_accuracy'])
    
    weights_checkpoint_file = weights_name.split('.')[0] + "-weights-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
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
            if name.find('weights-improvement') >= 0:
                val = int(name.split('.')[1])
                if val > max_val:
                    max_val = val
                    max_i = i
    
    if max_i == -1:
        best_weights = os.path.join(weights_path, weights_name)
    else:
        best_weights = os.path.join(weights_path, weights_names[max_i])

    project['number_categories'] = len(img_classes)
    project['learning_rate'] = project['learning_rate'] / project['learning_rate_modifier']
    project['best_weights'] = best_weights
    project['last_weights'] = os.path.join(weights_path, weights_name)

    return project