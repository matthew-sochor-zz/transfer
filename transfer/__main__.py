import sys
import argparse
import os

import yaml

from transfer.split import split_all
from transfer.project import configure, select_project, update_config
from transfer import images_to_array, pre_model
from transfer.model import train_model, predict_model

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
        
        if project['is_pre_model'] == False:
            project = pre_model(project)
            update_config(project)
        
        project = train_model(project)
        update_config(project)

        print('')
        print('Completed modeling round: ', project['model_round'])
        print('Best current model: ', project['best_weights'])
        print('Last current model: ', project['last_weights'])
        print('')
        print('To further refine the model, run again with:')
        print('')
        print('    transfer --run')
        print('')

    elif args.best_predict is not None:
        if project['best_weights'] is not None:
            if args.best_predict == 'default':
                args.best_predict = project['path']
            print('Predicting from current best model: ', project['best_weights'])
            print('Predicting on image(s) in: ', args.best_predict)
            predict_model(project, 'best_weights', args.best_predict)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print('    transfer --run')
            print('')

    elif args.last_predict is not None:
        if project['last_weights'] is not None:
            if args.last_predict == 'default':
                args.last_predict = project['path']
            print('Predicting from current last model: ', project['last_weights'])
            print('Predicting on image(s) in: ', args.last_predict)
            predict_model(project, 'last_weights', args.last_predict)
        else:
            print('Model is not trained.  Please first run your project:')
            print('')
            print('    transfer --run')
            print('')

    
