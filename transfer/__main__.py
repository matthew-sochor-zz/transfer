import sys
import argparse
import os

import yaml

from transfer.split import split_all
from transfer.project import configure, select_project, update_config
from transfer import images_to_array, pre_model
from transfer.model import train_model

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
            split_all(project['path'], project['test_percent'])
            project['is_split'] = True
            update_config(project)
        
        if project['is_array'] == False:
            images_to_array(project['path'], project['img_size'])
            project['is_array'] = True
            update_config(project)
        
        if project['is_pre_model'] == False:
            pre_model(project['path'], project['img_size'])
            project['is_pre_model'] = True
            update_config(project)
        
        best_model = train_model(project)
        project['model_round'] += 1
        project['learning_rate'] = project['learning_rate'] / project['learning_rate_modifier']
        project['model'] = best_model
        update_config(project)

    
