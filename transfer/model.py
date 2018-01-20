import os
import shutil
from subprocess import call

import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.inception_v3 import preprocess_input as vgg_preprocess_input
from keras import layers
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init
from termcolor import colored

from transfer.resnet50 import get_resnet_pre_post_model, get_resnet_final_model
from transfer.xception import get_xception_pre_post_model, get_xception_final_model
from transfer.inception_v3 import get_inception_v3_pre_post_model, get_inception_v3_final_model

def gen_minibatches(array_dir, array_names, batch_size, architecture, final = False):

    array_names = list(array_names)

    while True:
        # in place shuffle
        np.random.shuffle(array_names)
        array_names_mb = array_names[:batch_size]

        arrays = []
        labels = []
        for array_name in array_names_mb:
            img_path = os.path.join(array_dir, array_name)
            array = np.load(img_path)
            if final:
                if architecture == 'resnet50':
                    array = np.squeeze(resnet_preprocess_input(array[np.newaxis].astype(np.float32)))
                elif architecture == 'xception':
                    array = np.squeeze(xception_preprocess_input(array[np.newaxis].astype(np.float32)))
                else:
                    array = np.squeeze(inception_v3_preprocess_input(array[np.newaxis].astype(np.float32)))

            arrays.append(array)
            labels.append(np.load(img_path.replace('-img-','-label-')))

        yield np.array(arrays), np.array(labels)


def no_folds_generator(pre_model_files):
    yield [i for i in range(len(pre_model_files))], -1


def train_model(project, final = False, last = False):
    weight_label = '-' + project['architecture'] + '-weights-'
    source_path = project['path']
    weights_path = os.path.join(source_path, 'weights')
    plot_path = os.path.join(source_path, 'plots')
    if last:
        weights = 'last_weights'
    else:
        weights = 'best_weights'

    if final:
        weight_label += '-final-'
        use_path = os.path.join(source_path, 'augmented')
    else:
        use_path = os.path.join(source_path, 'pre_model')

    project['model_round'] += 1
#    call(['mkdir', '-p', weights_path])
    shutil.rmtree(weights_path,ignore_errors=True)
#    call(['mkdir', '-p', plot_path])
    os.makedirs(plot_path)

    img_dim = project['img_dim'] * project['img_size']
    conv_dim = project['conv_dim'] * project['img_size']

    lr =  project['learning_rate']
    decay = project['learning_rate_decay']

    all_files = os.listdir(use_path)
    pre_model_files = list(filter(lambda x: r'-img-' in x, all_files))
    label_names = list(filter(lambda x: r'-label-' in x, all_files))

    pre_model_files_df = pd.DataFrame({'files': pre_model_files})
    pre_model_files_df['suffix'] = pre_model_files_df.apply(lambda row: row.files.split('.')[-1], axis = 1)
    pre_model_files_df = pre_model_files_df[pre_model_files_df.suffix == 'npy']
    pre_model_files_df['ind'] = pre_model_files_df.apply(lambda row: row.files.split('-')[0], axis = 1).astype(int)
    pre_model_files_df['label'] = pre_model_files_df.apply(lambda row: row.files.split('-')[3], axis = 1)

    pre_model_files_df_dedup = pre_model_files_df.drop_duplicates(subset='ind')
    pre_model_files_df = pre_model_files_df.set_index(['ind'])

    pre_model_files.sort()
    label_names.sort()

    pre_model_files_arr = np.array(pre_model_files)
    label_names_arr = np.array(label_names)

    labels = [np.argmax(np.load(os.path.join(use_path, label_name))) for label_name in label_names]
    best_weights = []
    last_weights = []

    if project['kfold'] >= 3:
        kfold = StratifiedKFold(n_splits=project['kfold'], shuffle=True, random_state = project['seed'])
        kfold_generator = kfold.split(pre_model_files_df_dedup, pre_model_files_df_dedup.label)
        validate = True
    else:
        print('Too few k-folds selected, fitting on all data')
        kfold_generator = no_folds_generator(pre_model_files_df_dedup)
        validate = False

    for i, (train, test) in enumerate(kfold_generator):
        if project['kfold_every']:
            print('----- Fitting Fold', i, '-----')
        elif i > 0:
            break


        weights_name = project['name'] + weight_label + '-kfold-' + str(i) + '-round-' + str(project['model_round']) +'.hdf5'
        plot_name = project['name'] + weight_label + '-kfold-' + str(i) + '-round-' + str(project['model_round']) +'.png'

        if project[weights] is None:
            fold_weights = None
        else:
            fold_weights = project[weights][i]
        if final:
            if project['architecture'] == 'resnet50':
                model = get_resnet_final_model(img_dim, conv_dim, project['number_categories'], fold_weights, project['is_final'])
            elif project['architecture'] == 'xception':
                model = get_xception_final_model(img_dim, conv_dim, project['number_categories'], fold_weights, project['is_final'])
            else:
                model = get_inception_v3_final_model(img_dim, conv_dim, project['number_categories'], fold_weights, project['is_final'])

            for i, layer in enumerate(model.layers[1].layers):
                if len(layer.trainable_weights) > 0:
                    if i < project['final_cutoff']:
                        mult = 0.01
                    else:
                        mult = 0.1
                    layer.learning_rate_multiplier = [mult for tw in layer.trainable_weights]

        else:
            if project['architecture'] == 'resnet50':
                pre_model, model = get_resnet_pre_post_model(img_dim,
                                                    conv_dim,
                                                    len(project['categories']),
                                                    model_weights = fold_weights)
            elif project['architecture'] == 'xception':
                pre_model, model = get_xception_pre_post_model(img_dim,
                                                    conv_dim,
                                                    len(project['categories']),
                                                    model_weights = fold_weights)
            else:
                pre_model, model = get_inception_v3_pre_post_model(img_dim,
                                                    conv_dim,
                                                    len(project['categories']),
                                                    model_weights = fold_weights)

        pre_model_files_dedup_train = pre_model_files_df_dedup.iloc[train]
        train_ind = list(set(pre_model_files_dedup_train.ind))
        pre_model_files_train = pre_model_files_df.loc[train_ind]

        gen_train = gen_minibatches(use_path, pre_model_files_train.files, project['batch_size'], project['architecture'], final = final)
        number_train_samples = len(pre_model_files_train)

        if validate:
            pre_model_files_dedup_test = pre_model_files_df_dedup.iloc[test]
            test_ind = list(set(pre_model_files_dedup_test.ind))
            pre_model_files_test = pre_model_files_df.loc[test_ind]

            gen_test = gen_minibatches(use_path, pre_model_files_test.files, project['batch_size'], project['architecture'], final = final)
            number_test_samples = len(pre_model_files_test)
            validation_steps = (number_test_samples // project['batch_size'])

            weights_checkpoint_file = weights_name.split('.')[0] + '-kfold-' + str(i) + "-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_path, weights_checkpoint_file),
                                        monitor='val_categorical_accuracy',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max')

            callbacks_list = [checkpoint]
        else:
            gen_test = None
            validation_steps = None
            callbacks_list = None


        steps_per_epoch = (number_train_samples // project['batch_size'])
        for j in range(project['rounds']):
            optimizer = Adam(lr = lr, decay = decay)

            model.compile(optimizer = optimizer,
                        loss = 'categorical_crossentropy',
                        metrics = ['categorical_accuracy'])

            model.fit_generator(gen_train,
                                steps_per_epoch = steps_per_epoch,
                                epochs = project['cycle'] * (j + 1),
                                verbose = 1,
                                validation_data = gen_test,
                                validation_steps = validation_steps,
                                initial_epoch = j * project['cycle'],
                                callbacks = callbacks_list)

        model.save_weights(os.path.join(weights_path, weights_name))
        last_weights.append(os.path.join(weights_path, weights_name))
        weights_names = os.listdir(weights_path)
        max_val = -1
        max_i = -1
        for j, name in enumerate(weights_names):
            if name.find(weights_name.split('.')[0]) >= 0:
                if (name.find(weight_label) >= 0) and (name.find('improvement') >= 0):
                    val = int(name.split('.')[1])
                    if val > max_val:
                        max_val = val
                        max_i = j
        if project['plot']:
            print('Plotting confusion matrix')

            if max_i == -1:
                print('Loading last weights:', os.path.join(weights_path, weights_name))
                model.load_weights(os.path.join(weights_path, weights_name))
            else:
                print('Loading best weights:', os.path.join(weights_path, weights_names[max_i]))
                model.load_weights(os.path.join(weights_path, weights_names[max_i]))
            best_predictions = []
            true_labels = []

            print('Predicting test files')
            if validate:
                use_files = pre_model_files_test.files
            else:
                use_files = pre_model_files_train.files
            for array_name in tqdm(use_files):
                img_path = os.path.join(use_path, array_name)
                img = np.load(img_path)
                if final:
                    if project['architecture'] == 'resnet50':
                        img = np.squeeze(resnet_preprocess_input(img[np.newaxis].astype(np.float32)))
                    elif project['architecture'] == 'xception':
                        img = np.squeeze(xception_preprocess_input(img[np.newaxis].astype(np.float32)))
                    else:
                        img = np.squeeze(inception_v3_preprocess_input(img[np.newaxis].astype(np.float32)))
                prediction = model.predict(img[np.newaxis])
                best_predictions.append(project['categories'][np.argmax(prediction)])
                true_label = np.load(img_path.replace('-img-','-label-'))
                true_labels.append(project['categories'][np.argmax(true_label)])

            cm = confusion_matrix(true_labels, best_predictions, project['categories'])
            plt.clf()
            sns.heatmap(pd.DataFrame(cm, project['categories'], project['categories']), annot = True, fmt = 'g')
            plt.xlabel('Actual')
            plt.xlabel('Predicted')
            plt.xticks(rotation = 45, fontsize = 8)
            plt.yticks(rotation = 45, fontsize = 8)
            plt.title('Confusion matrix for fold: ' + str(i) + '\nweights' + weights_name)
            plt.savefig(os.path.join(plot_path, plot_name))
            print('Confusion matrix plot saved:', colored(os.path.join(plot_path, plot_name), 'magenta'))


        if max_i == -1:
            best_weights.append(os.path.join(weights_path, weights_name))
        else:
            best_weights.append(os.path.join(weights_path, weights_names[max_i]))

    project['number_categories'] = len(project['categories'])
    project['best_weights'] = best_weights
    project['last_weights'] = last_weights
    project['is_final'] = final

    return project
