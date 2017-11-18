import os
import re

from colorama import init
from termcolor import colored

def int_input(message, low, high, show_range = True):
    '''
    Ask a user for a int input between two values

    args:
        message (str): Prompt for user
        low (int): Low value, user entered value must be > this value to be accepted
        high (int): High value, user entered value must be < this value to be accepted
        show_range (boolean, Default True): Print hint to user the range

    returns:
        int_in (int): Input integer
    '''

    int_in = low - 1
    while (int_in < low) or (int_in > high):
        if show_range:
            suffix = ' (integer between ' + str(low) + ' and ' + str(high) + ')'
        else:
            suffix = ''
        inp = input('Enter a ' + message + suffix + ': ')
        if re.match('^[0-9]+$', inp) is not None:
            int_in = int(inp)
        else:
            print(colored('Must be an integer, try again!', 'red'))
    return int_in


def float_input(message, low, high):
    '''
    Ask a user for a float input between two values

    args:
        message (str): Prompt for user
        low (float): Low value, user entered value must be > this value to be accepted
        high (float): High value, user entered value must be < this value to be accepted

    returns:
        float_in (int): Input float
    '''

    float_in = low - 1.0
    while (float_in < low) or (float_in > high):
        inp = input('Enter a ' + message + ' (float between ' + str(low) + ' and ' + str(high) + '): ')
        if re.match('^([0-9]*[.])?[0-9]+$', inp) is not None:
            float_in = float(inp)
        else:
            print(colored('Must be a float, try again!', 'red'))
    return float_in

def bool_input(message):
    '''
    Ask a user for a boolean input

    args:
        message (str): Prompt for user

    returns:
        bool_in (boolean): Input boolean
    '''

    while True:
        suffix = ' (true or false): '
        inp = input(message + suffix)
        if inp.lower() == 'true':
            return True
        elif inp.lower() == 'false':
            return False
        else:
            print(colored('Must be either true or false, try again!', 'red'))

def str_input(message, inputs = None):

    user_str = None
    while user_str is None:
        inp = input(message)
        if inputs is None:
            user_str = inp
        elif inp in inputs:
            user_str = inp
        else:
            print(colored('Invalid input, should be one of:', 'red'))
            print(inputs)
    return user_str

def model_input(project):

    print('Select model weights:')
    print('[0] resnet best weights: ', colored(os.path.split(project['resnet_best_weights'])[-1], 'cyan'))
    print('[1] resnet last weights: ', colored(os.path.split(project['resnet_last_weights'])[-1], 'cyan'))
    print('[2] resnet w/ extra conv best weights: ', colored(os.path.split(project['extra_best_weights'])[-1], 'cyan'))
    print('[3] resnet w/ extra conv last weights: ', colored(os.path.split(project['extra_last_weights'])[-1], 'cyan'))
    model_choice = int_input('choice', 0, 3, show_range = False)
    weights = ['resnet_best_weights', 'resnet_last_weights', 'extra_best_weights', 'extra_last_weights'][model_choice]
    if model_choice < 2:
        extra_conv = False
    else:
        extra_conv = True
    return weights, extra_conv
