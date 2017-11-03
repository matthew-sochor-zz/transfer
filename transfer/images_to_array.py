import os
from subprocess import call

from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import load_img


def images_to_array(project):

    img_dim = 224 * project['img_size']
    print('Converting test images to array')
    val_images_to_array(project['img_path'], project['path'], 'test', img_dim)
    print('Converting train images to array')
    val_images_to_array(project['img_path'], project['path'], 'train', img_dim)

    project['is_array'] = True
    return project


def val_images_to_array(img_path, source_path, val_group, img_dim):

    split_path = os.path.join(source_path, 'split', val_group)
    array_path = os.path.join(source_path, 'array', val_group)
    call(['rm', '-rf', array_path])
    call(['mkdir', '-p', array_path])

    categories = sorted([d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))])

    print('Iterating over all categories: ', categories)

    for category_idx, category in enumerate(categories):
        print('categories:', category)
        category_path = os.path.join(split_path, category)
        img_files = sorted(os.listdir(category_path))
        for img_idx, img_file in tqdm(enumerate(img_files)):
            img_path = os.path.join(category_path, img_file)
            img = load_img(img_path, target_size=(img_dim, img_dim))

            img_name = '{}-img-{}-{}'.format(img_idx, category, category_idx)
            label_name = '{}-label-{}-{}'.format(img_idx, category, category_idx)

            label = np.eye(len(categories), dtype=np.float32)[category_idx]

            img_array_path = os.path.join(array_path, img_name)
            img_label_path = os.path.join(array_path, label_name)

            np.save(img_array_path, img)
            np.save(img_label_path, label)
