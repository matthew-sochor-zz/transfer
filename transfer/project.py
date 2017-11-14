import os
from subprocess import call

import yaml
import numpy as np
from colorama import init
from termcolor import colored

from transfer.input import int_input, float_input, bool_input, str_input


def configure():
    '''
    Configure the transfer environment and store
    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            config = yaml.load(fp.read())
    else:
        config = []

    project_name = input('Name your project: ')
    existing_project = None
    for project in config:
        if project_name == project['name']:
            existing_project = project_name
    if existing_project is not None:
        print(colored('Project ' + project_name + ' already exists', 'red'))
        overwrite = str_input('Would you like to overwrite this project? (yes or no) ', ['yes', 'no'])
        if overwrite == 'no':
            return
        else:
            config = [project for project in config if project_name != project['name']]

    image_path = os.path.expanduser(input('Select parent directory for your images: '))
    path_unset = True
    while path_unset:
        project_path = os.path.expanduser(input('Select destination for your project: '))
        if (project_path.find(image_path) == 0):
            print('Project destination should not be same or within image directory!')
        else:
            path_unset = False

    image_path = image_path.replace('~', home)
    test_percent = int_input('percentage of trips to assign to test (suggested: 20)', 1, 50)
    batch_size = int_input('batch size (suggested: 8)', 1, 64)
    learning_rate = float_input('learning rate (suggested: 0.001)', 0, 1)
    learning_rate_modifier = int_input('learing rate modifier (suggested: 5)', 1, 100)
    num_epochs = int_input('number of epochs (suggested: 10)', 1, 100)
    print('Select image resolution:')
    print('[0] low (224 px)')
    print('[1] mid (448 px)')
    print('[2] high (896 px)')
    img_resolution_index = int_input('choice', 0, 2, show_range = False)
    if img_resolution_index == 0:
        img_size = 1
    elif img_resolution_index == 1:
        img_size = 2
    else:
        img_size = 4
    use_augmentation = str_input('Would you like to add image augmentation? (yes or no) ', ['yes', 'no'])
    if use_augmentation == 'yes':
        augmentations = select_augmentations()
    else:
        augmentations = None

    project = {'name': project_name,
               'img_path': image_path,
               'path': project_path,
               'test_percent': test_percent,
               'batch_size': batch_size,
               'resnet_learning_rate': learning_rate,
               'extra_learning_rate': learning_rate,
               'resnet_learning_rate_modifier': learning_rate_modifier,
               'extra_learning_rate_modifier': learning_rate_modifier,
               'epochs': num_epochs,
               'img_size': img_size,
               'augmentations': augmentations,
               'is_split': False,
               'is_array': False,
               'is_augmented': False,
               'is_pre_model': False,
               'model_round': 0,
               'resnet_last_weights': None,
               'extra_last_weights': None,
               'resnet_best_weights': None,
               'extra_best_weights': None,
               'seed': None}

    config.append(project)
    store_config(config)
    print('')
    print(colored('Project configure saved!', 'cyan'))
    print('')
    print('To run project:')
    print('')
    print(colored('    transfer --run --project ' + project_name, 'green'))
    print('or')
    print(colored('    transfer -r -p ' + project_name, 'green'))


def select_augmentations():
    print('Select augmentations:')
    print(colored('Note: defaults are all zero or false.', 'cyan'))
    rounds = int_input('number of augmentation rounds', 1, 100)
    featurewise_center = bool_input('featurewise_center: set input mean to 0 over the dataset.')
    featurewise_std_normalization = bool_input('featurewise_std_normalization: divide inputs by std of the dataset.')
    samplewise_center = bool_input('samplewise_center: set each sample mean to 0.')
    samplewise_std_normalization = bool_input('samplewise_std_normalization: divide each input by its std.')
    zca_whitening = bool_input('zca_whitening: apply ZCA whitening.')
    rotation_range = int_input('rotation_range: degrees', 0, 180, True)
    width_shift_range = float_input('width_shift_range: fraction of total width.', 0., 1.)
    height_shift_range = float_input('height_shift_range: fraction of total width.', 0., 1.)
    shear_range = float_input('shear_range: shear intensity (shear angle in radians)', 0., np.pi/2)
    zoom_range = float_input('zoom_range: amount of zoom. Zoom will be randomly picked in the range [1-z, 1+z].', 0., 1.)
    channel_shift_range = float_input('channel_shift_rangee: shift range for each channels.', 0., 1.)
    print('fill_mode: points outside the boundaries are filled according to the given mode.')
    fill_mode = str_input('constant, nearest, reflect, or wrap. Default nearest: ',['constant', 'nearest', 'reflect', 'wrap'])
    if (fill_mode == 'constant'):
        cval = float_input('cval: value used for points outside the boundaries', 0., 1.)
    else:
        cval = 0.0
    horizontal_flip = bool_input('horizontal_flip: whether to randomly flip images horizontally.')
    vertical_flip = bool_input('vertical_flip: whether to randomly flip images vertically.')
    rescale = int_input('rescale: rescaling factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.', 0, 255)
    if rescale == 0:
        rescale = None

    augmentations = {'rounds': rounds,
                     'featurewise_center': featurewise_center,
                     'featurewise_std_normalization': featurewise_std_normalization,
                     'samplewise_center': samplewise_center,
                     'samplewise_std_normalization': samplewise_std_normalization,
                     'zca_whitening': zca_whitening,
                     'rotation_range': rotation_range,
                     'width_shift_range': width_shift_range,
                     'height_shift_range': height_shift_range,
                     'shear_range': shear_range,
                     'zoom_range': zoom_range,
                     'channel_shift_range': channel_shift_range,
                     'fill_mode': fill_mode,
                     'cval': cval,
                     'horizontal_flip': horizontal_flip,
                     'vertical_flip': vertical_flip,
                     'rescale': rescale}
    return augmentations


def select_project(user_provided_project):
    '''
    Select a project from configuration to run transfer on

    args:
        user_provided_project (str): Project name that should match a project in the config

    returns:
        project (dict): Configuration settings for a user selected project

    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())
        if len(projects) == 1:
            project = projects[0]
        else:
            if user_provided_project in [project['name'] for project in projects]:
                for inner_project in projects:
                    if user_provided_project == inner_project['name']:
                        project = inner_project
            else:
                print('Select your project')
                for i, project in enumerate(projects):
                    print('[' + str(i) + ']: ' + project['name'])
                project_index = int_input('project', -1, len(projects), show_range = False)
                project = projects[project_index]
    else:
        print('Transfer is not configured.')
        print('Please run:')
        print('')
        print(colored('    transfer --configure', 'green'))
        return

    print(colored('Project selected: ' + project['name'], 'cyan'))
    return project

def store_config(config):
    '''
    Store configuration

    args:
        config (list[dict]): configurations for each project
    '''
    home = os.path.expanduser('~')

    call(['mkdir', '-p', os.path.join(home, '.transfer')])
    with open(os.path.join(home, '.transfer', 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)


def update_config(updated_project):
    '''
    Update project in configuration

    args:
        updated_project (dict): Updated project configuration values

    '''

    home = os.path.expanduser('~')
    if os.path.isfile(os.path.join(home, '.transfer', 'config.yaml')):
        with open(os.path.join(home, '.transfer', 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())
        replace_index = -1
        for i, project in enumerate(projects):
            if project['name'] == updated_project['name']:
                replace_index = i

        if replace_index > -1:
            projects[replace_index] = updated_project
            store_config(projects)
        else:
            print('Not saving configuration')
            print(colored('Project: ' + updated_project['name'] + ' was not found in configured projects!', 'red'))

    else:
        print('Transfer is not configured.')
        print('Please run:')
        print('')
        print(colored('    transfer --configure', 'cyan'))
        return
