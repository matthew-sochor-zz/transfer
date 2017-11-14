import sys
import argparse
import os

import yaml
import keras
from colorama import init
from termcolor import colored

from transfer.split import split_all
from transfer.project import configure, select_project, update_config
from transfer import images_to_array, pre_model
from transfer.model import train_model
from transfer.predict_model import predict_model, predict_activation_model
from transfer.augment_arrays import augment_arrays

def main(args = None):
    '''
    Main entry point for transfer command line tool.

    This essentially will marshall the user to the functions they need.
    '''

    parser = argparse.ArgumentParser(description = 'Tool to perform transfer learning')

    parser.add_argument('-c','--configure',
                        action = 'store_true',
                        help = 'Configure transfer')

    parser.add_argument('-p','--project',
                        action = 'store',
                        type = str,
                        default = None,
                        dest = 'project',
                        help = 'Split files into test and train')

    parser.add_argument('-r','--run',
                        action = 'store_true',
                        help = 'Run all transfer learning operations')

    parser.add_argument('--best-predict',
                        action = 'store',
                        type = str,
                        default = None,
                        const = 'default',
                        dest = 'best_predict',
                        nargs='?',
                        help = 'Predict best model on directory')

    parser.add_argument('--last-predict',
                        action = 'store',
                        type = str,
                        default = None,
                        const = 'default',
                        dest = 'last_predict',
                        nargs='?',
                        help = 'Predict last model on directory')

    parser.add_argument('--best-predict-activation',
                        action = 'store',
                        type = str,
                        default = None,
                        const = 'default',
                        dest = 'best_predict_activation',
                        nargs='?',
                        help = 'Predict best activation model on directory')

    parser.add_argument('--last-predict-activation',
                        action = 'store',
                        type = str,
                        default = None,
                        const = 'default',
                        dest = 'last_predict_activation',
                        nargs='?',
                        help = 'Predict last activation model on directory')
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.configure:
        configure()
        return
    else:
        project = select_project(args.project)

    if args.run:
        if project['is_split'] == False:
            project = split_all(project)
            update_config(project)

        if project['is_array'] == False:
            project = images_to_array(project)
            update_config(project)

        if project['is_augmented'] == False:
            project = augment_arrays(project)
            update_config(project)

        if project['is_pre_model'] == False:
            project = pre_model(project)
            update_config(project)

        project = train_model(project)
        update_config(project)

        project = train_model(project, extra_conv = True)
        update_config(project)

        print('')
        print(colored('Completed modeling round: ' + project['model_round'], 'cyan'))
        print('')
        print('Best current model: ', colored(project['resnet_best_weights'], 'yellow'))
        print('Last current model: ', colored(project['resnet_last_weights'], 'yellow'))
        print('')
        print('To further refine the model, run again with:')
        print('')
        print(colored('    transfer --run', 'green'))
        print('')

    elif args.best_predict is not None:
        if project['resnet_best_weights'] is not None:
            if args.best_predict == 'default':
                args.best_predict = project['img_path']
            print('Predicting from current best model: ', colored(project['resnet_best_weights'], 'yellow'))
            print('Predicting on image(s) in: ', colored(args.best_predict, 'yellow'))
            predict_model(project, 'resnet_best_weights', args.best_predict)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print(colored('    transfer --run', 'green'))
            print('')

    elif args.last_predict is not None:
        if project['resnet_last_weights'] is not None:
            if args.last_predict == 'default':
                args.last_predict = project['img_path']
            print('Predicting from current last model: ', colored(project['resnet_last_weights'], 'yellow'))
            print('Predicting on image(s) in: ', colored(args.last_predict, 'yellow'))
            predict_model(project, 'resnet_last_weights', args.last_predict)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print(colored('    transfer --run', 'green'))
            print('')


    elif args.best_predict_activation is not None:
        keras.backend.set_learning_phase(0)
        if project['extra_best_weights'] is not None:
            if args.best_predict_activation == 'default':
                args.best_predict_activation = project['img_path']
            print('Predicting activation from current best model: ', colored(project['extra_best_weights'], 'yellow'))
            print('Predicting on image(s) in: ', colored(args.best_predict_activation, 'yellow'))
            predict_activation_model(project, 'extra_best_weights', args.best_predict_activation)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print(colored('    transfer --run', 'green'))
            print('')

    elif args.last_predict_activation is not None:
        keras.backend.set_learning_phase(0)
        if project['extra_last_weights'] is not None:
            if args.last_predict_activation == 'default':
                args.last_predict_activation = project['img_path']
            print('Predicting activation from current last model: ', colored(project['extra_last_weights'], 'yellow'))
            print('Predicting on image(s) in: ', colored(args.last_predict_activation, 'yellow'))
            predict_activation_model(project, 'extra_last_weights', args.last_predict_activation)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print(colored('    transfer --run', 'green'))
            print('')
