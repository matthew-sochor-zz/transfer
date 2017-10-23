import os
from subprocess import call

import yaml

from transfer.input import int_input, float_input


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
    image_path = input('Select parent directory for your images: ')
    image_path = image_path.replace('~', home)
    test_percent = int_input('percentage of trips to assign to test (suggested: 20)', 0, 100)
    batch_size = int_input('batch size (suggested: 8)', 0, 65)
    learning_rate = float_input('learning rate (suggested: 0.001)', 0, 1)
    learning_rate_modifier = int_input('learing rate modifier (suggested: 5)', 1, 100)
    num_epochs = int_input('number of epochs (suggested: 10)', 1, 100)
    print('Select image resolution:')
    print('[0] low (224 px)')
    print('[1] mid (448 px)')
    print('[2] high (896 px)')
    img_resolution_index = int_input('choice', -1, 2, show_range = False)
    if img_resolution_index == 0:
        img_size = 1
    elif img_resolution_index == 1:
        img_size = 2
    else:
        img_size = 4
        
    project = {'name': project_name,
               'path': image_path,
               'test_percent': test_percent,
               'batch_size': batch_size,
               'learning_rate': learning_rate,
               'learning_rate_modifier': learning_rate_modifier,
               'epochs': num_epochs,
               'img_size': img_size,
               'is_split': False,
               'is_array': False,
               'is_augmented': False,
               'is_pre_model': False,
               'model_round': 1,
               'model': None,
               'mod_model': None}
               
    config.append(project)
    store_config(config)


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
        print('transfer --configure')
        return

    print('Project selected: ', project['name'])
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
            print('Project: ', updated_project['name'], 'was not found in configured projects!')

    else:
        print('Transfer is not configured.')
        print('Please run:')
        print('')
        print('transfer --configure')
        return
