import os
import shutil
import tarfile
import readline
import glob

import yaml
import numpy as np
from colorama import init
from termcolor import colored

from transfer.input import int_input, float_input, bool_input, str_input


class Completer(object):
    def path_completer(self, text, state):
        return [os.path.join(x, '') if os.path.isdir(x) else x for x in glob.glob(os.path.expanduser(text) + '*')][state]


def configure():
    '''
    Configure the transfer environment and store
    '''
    completer = Completer()
    readline.set_completer_delims('\t')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(completer.path_completer)

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

    print('Select architecture:')
    print('[0] resnet50')
    print('[1] xception')
    print('[2] inception_v3')
    architecture = int_input('choice', 0, 2, show_range = False)
    if architecture == 0:
        arch = 'resnet50'
        img_dim = 224
        conv_dim = 7
        final_cutoff = 80
    elif architecture == 1:
        arch = 'xception'
        img_dim = 299
        conv_dim = 10
        final_cutoff = 80
    else:
        arch = 'inception_v3'
        img_dim = 299
        conv_dim = 8
        final_cutoff = 80
    api_port = int_input('port for local prediction API (suggested: 5000)', 1024, 49151)
    kfold = int_input('number of folds to use (suggested: 5)', 3, 10)
    kfold_every = bool_input('Fit a model for every fold? (if false, just fit one)')
    print('Warning: if working on a remote computer, you may not be able to plot!')
    plot_cm = bool_input('Plot a confusion matrix after training?')
    batch_size = int_input('batch size (suggested: 8)', 1, 64)
    learning_rate = float_input('learning rate (suggested: 0.001)', 0, 1)
    learning_rate_decay = float_input('learning decay rate (suggested: 0.000001)', 0, 1)
    cycle = int_input('number of cycles before resetting the learning rate (suggested: 3)', 1, 10)
    num_rounds = int_input('number of rounds (suggested: 3)', 1, 100)
    print('Select image resolution:')
    print('[0] low (' + str(img_dim) + ' px)')
    print('[1] mid (' + str(img_dim * 2) + ' px)')
    print('[2] high (' + str(img_dim * 4) + ' px)')
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
               'plot': plot_cm,
               'api_port': api_port,
               'kfold': kfold,
               'kfold_every': kfold_every,
               'cycle': cycle,
               'seed': np.random.randint(9999),
               'batch_size': batch_size,
               'learning_rate': learning_rate,
               'learning_rate_decay': learning_rate_decay,
               'final_cutoff': final_cutoff,
               'rounds': num_rounds,
               'img_size': img_size,
               'augmentations': augmentations,
               'architecture': arch,
               'img_dim': img_dim,
               'conv_dim': conv_dim,
               'is_split': False,
               'is_array': False,
               'is_augmented': False,
               'is_pre_model': False,
               'is_final': False,
               'model_round': 0,
               'server_weights': None,
               'last_weights': None,
               'best_weights': None}

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

    project = {'name': project_name,
               'api_port': api_port,
               'img_size': img_size,
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
    # These three are not implemented because they require training and then that would
    #  need to get propogated over which is complicated for prediction

    featurewise_center = False #bool_input('featurewise_center: set input mean to 0 over the dataset.')
    featurewise_std_normalization = False #bool_input('featurewise_std_normalization: divide inputs by std of the dataset.')
    zca_whitening = False #bool_input('zca_whitening: apply ZCA whitening.')
    samplewise_center = False #bool_input('samplewise_center: set each sample mean to 0.')
    samplewise_std_normalization = False #bool_input('samplewise_std_normalization: divide each input by its std.')
    rotation_range = int_input('rotation_range: degrees', 0, 180, True)
    width_shift_range = float_input('width_shift_range: fraction of total width.', 0., 1.)
    height_shift_range = float_input('height_shift_range: fraction of total width.', 0., 1.)
    shear_range = float_input('shear_range: shear intensity (shear angle in radians)', 0., np.pi/2)
    zoom_range_in = float_input('zoom_range: amount of zoom in. 1.0 is no zoom, 0 is full zoom.', 0., 1.)
    zoom_range_out = float_input('zoom_range: amount of zoom out. 1.0 is no zoom, 2.0 is full zoom ', 1., 2.)
    channel_shift_range = float_input('channel_shift_rangee: shift range for each channels.', 0., 1.)
    print('fill_mode: points outside the boundaries are filled according to the given mode.')
    fill_mode = str_input('constant, nearest, reflect, or wrap. Default nearest: ',['constant', 'nearest', 'reflect', 'wrap'])
    if (fill_mode == 'constant'):
        cval = float_input('cval: value used for points outside the boundaries', 0., 1.)
    else:
        cval = 0.0
    horizontal_flip = bool_input('horizontal_flip: whether to randomly flip images horizontally.')
    vertical_flip = bool_input('vertical_flip: whether to randomly flip images vertically.')
    rescale = None #int_input('rescale: rescaling factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.', 0, 255)
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
                     'zoom_range': [zoom_range_in, zoom_range_out],
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


def read_imported_config(project_path, project_name, projects = None):

    completer = Completer()
    readline.set_completer_delims('\t')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(completer.path_completer)

    # Oh god this logic is a disaster, user interfaces are hard
    bad_user = True
    while bad_user:
        relearn_str = str_input('Do you want to learn on new starting from these weights? (yes or no) ')
        if relearn_str.lower() == 'yes' or relearn_str.lower() == 'y':
            bad_user = False
            relearn = True
        elif relearn_str.lower() == 'no' or relearn_str.lower() == 'n':
            bad_user = False
            relearn = False

    unique_name = False
    while unique_name == False:
        unique_name = True
        if projects is not None:
            for project in projects:
                if project['name'] == project_name:
                    print(colored('Project with this name already exists.', 'red'))
                    project_name = str_input('Provide a new project name: ')
                    unique_name = False
    if relearn:
        image_path = os.path.expanduser(input('Select parent directory for your images: '))
        path_unset = True
        while path_unset:
            project_dest = os.path.expanduser(input('Select destination for your project: '))
            if (project_dest.find(image_path) == 0):
                print('Project destination should not be same or within image directory!')
            else:
                path_unset = False
    else:
        project_dest = os.path.expanduser(input('Select destination for your project: '))

    if os.path.isdir(project_dest) == False:
        print('Creating directory:', project_dest)
        os.makedirs(project_dest, exist_ok = True)
    # You don't get to judge me t('-' t)
    with open(os.path.join(project_path, 'config.yaml'), 'r') as fp:
        import_project = yaml.load(fp.read())
    import_project['name'] = project_name
    import_project['path'] = project_dest

    if relearn:

        kfold = int_input('number of folds to use (suggested: 5)', 3, 10)
        kfold_every = bool_input('Fit a model for every fold? (if false, just fit one)')
        print('Warning: if working on a remote computer, you may not be able to plot!')
        plot_cm = bool_input('Plot a confusion matrix after training?')
        batch_size = int_input('batch size (suggested: 8)', 1, 64)
        learning_rate = float_input('learning rate (suggested: 0.001)', 0, 1)
        learning_rate_decay = float_input('learning decay rate (suggested: 0.000001)', 0, 1)
        cycle = int_input('number of cycles before resetting the learning rate (suggested: 3)', 1, 10)
        num_rounds = int_input('number of rounds (suggested: 3)', 1, 100)

        import_project['img_path'] = image_path
        import_project['best_weights'] = [os.path.join(project_path, weight) for weight in os.listdir(project_path) if weight.find('.hdf5') > 0]
        import_project['last_weights'] = import_project['best_weights']
        import_project['server_weights'] = None
        import_project['kfold'] = kfold
        import_project['kfold_every'] = kfold_every
        import_project['cycle'] = cycle
        import_project['seed'] = np.random.randint(9999)
        import_project['batch_size'] = batch_size
        import_project['learning_rate'] = learning_rate
        import_project['learning_rate_decay'] = learning_rate_decay
        if 'final_cutoff' not in import_project.keys():
            import_project['final_cutoff'] = 80
        import_project['rounds'] = num_rounds
        import_project['is_split'] = False
        import_project['is_array'] = False
        import_project['is_augmented'] = False
        import_project['is_pre_model'] = False
        import_project['model_round'] = 1
        import_project['plot'] = plot_cm

        print('')
        print('To re-learn new images with project:')
        print('')
        print(colored('    transfer --run --project ' + project_name, 'green'))
        print('or')
        print(colored('    transfer -r -p ' + project_name, 'green'))
        print('')
    else:
        import_project['server_weights'] = [os.path.join(project_path, weight) for weight in os.listdir(project_path) if weight.find('.hdf5') > 0]

    return import_project

def import_config(config_file):

    config_file = os.path.expanduser(config_file)
    print(config_file)
    transfer_path = os.path.expanduser(os.path.join('~','.transfer'))

    import_temp_path = os.path.join(transfer_path, 'import-temp')
    import_path = os.path.join(transfer_path, 'import')
    shutil.rmtree(import_temp_path, ignore_errors = True)
    os.makedirs(import_temp_path, exist_ok = True)

    if os.path.isfile(config_file) == False:
        print('This is not a file:', colored(config_file, 'red'))
        return

    with tarfile.open(config_file, mode = "r:gz") as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path=import_temp_path)

    for listed in os.listdir(import_temp_path):
        if os.path.isdir(os.path.join(import_temp_path, listed)):
            project_name = listed

    project_path = os.path.join(transfer_path, 'import', project_name)
    shutil.rmtree(project_path, ignore_errors = True)
    os.makedirs(os.path.join(transfer_path, 'import'), exist_ok = True)

    shutil.move(os.path.join(import_temp_path, project_name), import_path)
    shutil.rmtree(import_temp_path, ignore_errors = True)

    print('Imported project:', colored(project_name, 'magenta'))
    if os.path.isfile(os.path.join(transfer_path, 'config.yaml')):
        with open(os.path.join(transfer_path, 'config.yaml'), 'r') as fp:
            projects = yaml.load(fp.read())

        import_project = read_imported_config(project_path, project_name, projects)

        projects.append(import_project)
        store_config(projects)

    else:
        shutil.copy(os.path.join(project_path, 'config.yaml'), os.path.join(transfer_path, 'config.yaml'))

        import_project = read_imported_config(project_path, project_name)

        store_config([import_project])

    print('Project successfully imported!')
    print('')
    print('Make predictions with:')
    print('')
    print(colored('    transfer --predict [optional dir or file] --project ' + import_project['name'], 'yellow'))
    print('')
    print('Or start a prediction server with:')
    print('')
    print(colored('    transfer --prediction-rest-api --project ' + import_project['name'], 'yellow'))


def export_config(config, weights, ind = None):
    export_path = os.path.expanduser(os.path.join('~','.transfer','export', config['name']))
    if ind is None:
        export_tar = export_path + '_' + weights + '.tar.gz'
    else:
        export_tar = export_path + '_' + weights + '_kfold_' + str(ind) + '.tar.gz'

    os.makedirs(export_path, exist_ok = True)
    server_weights = []
    if ind is None:
        for i in range(len(config[weights])):
            server_weights.append(os.path.join(export_path, 'server_model_kfold_' + str(i) +'.hdf5'))
            shutil.copy(config[weights][i], server_weights[-1])
    else:
        server_weights = [os.path.join(export_path, 'server_model_kfold_' + str(ind) +'.hdf5')]
        shutil.copy(config[weights][ind], server_weights[-1])

    project = {'name': config['name'],
               'api_port': config['api_port'],
               'img_size': config['img_size'],
               'img_dim': config['img_dim'],
               'conv_dim': config['conv_dim'],
               'final_cutoff': config['final_cutoff'],
               'architecture': config['architecture'],
               'number_categories': config['number_categories'],
               'categories': config['categories'],
               'augmentations': config['augmentations'],
               'is_final': config['is_final'],
               'server_weights': server_weights}
    store_config(project, suffix = os.path.join('export', config['name']))

    with tarfile.open(export_tar, mode = "w:gz") as tf:
        tf.add(os.path.expanduser(os.path.join('~', '.transfer', 'export', config['name'])), config['name'])

    shutil.rmtree(export_path, ignore_errors = True)
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

    os.makedirs(config_path, exist_ok = True)
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
