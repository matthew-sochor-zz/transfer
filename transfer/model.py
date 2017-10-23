import os
from subprocess import call

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import layers

from transfer.resnet50 import get_post_model

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


def train_model(project):

    model_name = project['name'] + '_' + str(project['model_round']) +'.hdf5'
    source_path = project['path']

    img_classes = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path,d))]

    model_path = os.path.join(source_path, '..', 'model')
    pre_model_path_test = os.path.join(source_path, '..', 'pre_model', 'test')
    pre_model_path_train = os.path.join(source_path, '..', 'pre_model', 'train')
    call(['mkdir', '-p', model_path])

    number_train_samples = len(os.listdir(pre_model_path_train))
    number_test_samples = len(os.listdir(pre_model_path_test))

    gen_train = gen_minibatches(pre_model_path_train, project['batch_size'])
    gen_test = gen_minibatches(pre_model_path_test, project['batch_size'])

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    model = get_post_model(img_dim, conv_dim, len(img_classes), model_weights = project['model'])

    optimizer = Adam(lr = project['learning_rate'])
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['categorical_accuracy'])
    
    model_checkpoint_file = model_name.split('.')[0] + "-weights-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
    filepath = os.path.join(model_path, model_checkpoint_file)
    checkpoint = ModelCheckpoint(filepath, 
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

    '''
    Y_test = []
    Y_pred = []
    for _, (x_test, y_test) in zip(range(nbr_tst_samples // batch_size), gen_tst):
        Y_test.append(y_test)
        Y_pred.append(model.predict_on_batch(x_test))

    print('Model test:', np.mean(np.argmax(np.concatenate(Y_test), axis=1) == np.argmax(np.concatenate(Y_pred), axis=1)))
    '''
    model.save(os.path.join(model_path, model_name))
    model_names = os.listdir(model_path)
    max_val = -1
    max_i = -1
    for i, name in enumerate(model_names):
        if name.find(model_name.split('.')[0]) >= 0:
            if name.find('weights-improvement') >= 0:
                val = int(name.split('.')[1])
                if val > max_val:
                    max_val = val
                    max_i = i
    if max_i == -1:
        return os.path.join(model_path, model_name)
    else:
        return os.path.join(model_path, model_names[max_i])