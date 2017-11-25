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

    api_port = int_input('port for local prediction API (suggested: 5000)', 1024, 49151)
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
               'api_port': api_port,
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
               'server_weights': None,
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


def configure_server():
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

    api_port = int_input('port for local prediction API (suggested: 5000)', 1024, 49151)
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
    num_categories = int_input('number of image categories in your model', 0, 10000000)

    weights = False
    while weights == False:
        server_weights = os.path.expanduser(input('Select weights file: '))
        if os.path.isfile(server_weights):
            weights = True
        else:
            print('Cannot find the weight file: ', server_weights)

    extra_conv = bool_input('Do the weights have the extra convolutional layer? ')

    project = {'name': project_name,
               'api_port': api_port,
               'img_size': img_size,
               'extra_conv': extra_conv,
               'number_categories': num_categories,
               'server_weights': server_weights}

    config.append(project)
    store_config(config)
    print('')
    print(colored('Project configure saved!', 'cyan'))
    print('')
    print('To start the server:')
    print('')
    print(colored('    transfer --prediction-rest-api --project ' + project_name, 'green'))
    print('or')
    print(colored('    transfer --prediction-rest-api -p ' + project_name, 'green'))


def select_augmentations():
    print('Select augmentations:')
    print(colored('Note: defaults are all zero or false.', 'cyan'))
    rounds = int_input('number of augmentation rounds', 1, 100)
    featurewise_center = False #bool_input('featurewise_center: set input mean to 0 over the dataset.')
    featurewise_std_normalization = False #bool_input('featurewise_std_normalization: divide inputs by std of the dataset.')
    samplewise_center = False #bool_input('samplewise_center: set each sample mean to 0.')
    samplewise_std_normalization = False #bool_input('samplewise_std_normalization: divide each input by its std.')
    zca_whitening = False #bool_input('zca_whitening: apply ZCA whitening.')
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


def read_imported_config(import_path, project_name, projects = None):

    # Oh god this logic is a disaster, user interfaces are hard
    unique_name = False
    project_path = os.path.join(import_path, project_name)
    while unique_name == False:
        unique_name = True
        if projects is not None:
            for project in projects:
                if project['name'] == project_name:
                    print(colored('Project with this name already exists.', 'red'))
                    project_name = str_input('Provide a new project name: ')
                    unique_name = False

    project_dest = os.path.expanduser(str_input('Provide a path for your predictions to be saved: '))
    if os.path.isdir(project_dest) == False:
        print('Creating directory:', project_dest)
        call(['mkdir', '-p', project_dest])
    # You don't get to judge me t('-' t)
    with open(os.path.join(project_path, 'config.yaml'), 'r') as fp:
        import_project = yaml.load(fp.read())
    import_project['name'] = project_name
    import_project['server_weights'] = os.path.join(project_path, 'server_model.hdf5')
    import_project['path'] = project_dest
    return import_project

def import_config(config_file):
    config_file = os.path.expanduser(config_file)
    transfer_path = os.path.expanduser(os.path.join('~/.transfer'))
    import_path = os.path.join(transfer_path, 'import')
    call(['rm', '-rf', import_path])
    call(['mkdir','-p', import_path])

    if os.path.isfile(config_file) == False:
        print('This is not a file:', colored(config_file, 'red'))
        return

    call(['tar', '-zxvf', config_file, '-C', import_path])
    project_name = os.listdir(import_path)[0]
    print('Imported project:', colored(project_name, 'magenta'))
    if os.path.isfile(os.path.join(transfer_path, 'config.yaml')):
        with open(os.path.join(transfer_path, 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())

        import_project = read_imported_config(import_path, project_name, projects)

        projects.append(import_project)
        store_config(projects)

    else:
        call(['cp', os.path.join(import_path, project_name, 'config.yaml'), os.path.join(transfer_path, 'config.yaml')git ])
        print(os.listdir(import_path))
        import_project = read_imported_config(import_path, project_name)
        store_config([import_project])

    print('Project successfully imported!')
    print('Make predictions with:')
    print('')
    print(colored('transfer --predict [optional dir or file] --project ' + import_project['name'], 'yellow'))
    print('')
    print('Or start a prediction server with:')
    print('')
    print(colored('transfer --prediction-rest-api --project ' + import_project['name'], 'yellow'))


def export_config(config, weights, extra_conv):
    export_path = os.path.expanduser(os.path.join('~/.transfer/export', config['name']))
    export_tar = export_path + '.tar.gz'
    call(['mkdir','-p', export_path])

    server_weights = os.path.join(export_path, 'server_model.hdf5')
    call(['cp', config[weights], server_weights])

    project = {'name': config['name'],
               'api_port': config['api_port'],
               'img_size': config['img_size'],
               'extra_conv': extra_conv,
               'number_categories': config['number_categories'],
               'categories': config['categories'],
               'augmentations': config['augmentations'],
               'server_weights': server_weights}
    store_config(project, suffix = os.path.join('export', config['name']))
    call(['tar', '-zcvf', export_tar, '-C', os.path.expanduser('~/.transfer/export'), config['name']])
    print('Project successfully exported, please save the following file for re-import to transfer')
    print('')
    print(colored(export_tar, 'green'))


def store_config(config, suffix = None):
    '''
    Store configuration

    args:
        config (list[dict]): configurations for each project
    '''
    home = os.path.expanduser('~')
    if suffix is not None:
        config_path = os.path.join(home, '.transfer', suffix)
    else:
        config_path = os.path.join(home, '.transfer')

    call(['mkdir', '-p', config_path])
    with open(os.path.join(config_path, 'config.yaml'), 'w') as fp:
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
