import os
import sys
from subprocess import call

import numpy as np
from tqdm import tqdm


def split_all(project):
    '''
    Randomly split files in test and train

    Args:
        source_path (str): system path where all images are located
        test_fraction (int, Default 10): Fraction of images to assign to test set (should be 0 < test_fraction < 100)
        seed (int, Default None): Seed for random splits

    Returns:
        project (dict): Project state
    '''

    source_path = project['img_path']
    dest_path = project['path']

    img_classes = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path,d))]
    test_path = make_paths(dest_path, 'test', img_classes)
    train_path = make_paths(dest_path, 'train', img_classes)

    for img_class in img_classes:
        split_group(source_path, img_class, test_path, train_path, project['test_percent'], project['seed'])

    project['is_split'] = True
    return project


def split_group(source_path, group_name, test_path, train_path, test_percent, seed):
    '''
    Randomly split files in test and train

    Args:
        source_path (str): system path where all images are located
        group_name (str): sub-path in source_path with a single image group
        test_path (str): location for test images to be saved
        train_path (str): location for train images to be saved
        test_fraction (int, Default 10): Fraction of images to assign to test set (should be 0 < test_fraction < 100)
        seed (int, Default None): Seed for random splits
    '''

    images = os.listdir(source_path + '/' + group_name)
    images = [f for f in images
              if (f.find('.jpg') > -1) or
                 (f.find('.jpeg') > -1) or
                 (f.find('.png') > -1)]

    index = int(len(images) * test_percent / 100.0)
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(images)
    test, train = images[:index], images[index:]

    print('Splitting test set of: ', group_name)
    for images in tqdm(test):
        call(['cp', os.path.join(source_path, group_name, images), os.path.join(test_path, group_name, images)])

    print('Splitting train set of: ', group_name)
    for images in tqdm(train):
        call(['cp', os.path.join(source_path, group_name, images), os.path.join(train_path, group_name, images)])


def make_paths(source_path, val_group, img_classes):
    '''
    Create split directory alongside source path and all validation and class subdirectories

    args:
        source_path (str): system path where all images are located
        val_group (str): test or train
        img_classes (list[str]): image classes

    returns:
        val_path (str): system path where all split images are located
    '''

    val_path = os.path.join(source_path, 'split', val_group)
    call(['rm', '-rf', val_path])
    call(['mkdir', '-p', val_path])
    for img_class in img_classes:
        call(['mkdir', '-p', os.path.join(val_path, img_class)])

    return val_path
