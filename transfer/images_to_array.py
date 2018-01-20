import os
import shutil
from subprocess import call

from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import load_img


def images_to_array(project):

    categories = [d for d in os.listdir(project['img_path']) if os.path.isdir(os.path.join(project['img_path'],d))]
    project['categories'] = categories
    img_dim = project['img_dim'] * project['img_size']
    print('Converting images to array')
    category_rounds = val_images_to_array(project['img_path'], project['path'], img_dim, project['categories'])

    project['is_array'] = True
    project['category_rounds'] = category_rounds
    return project


def val_images_to_array(img_path, source_path, img_dim, categories):

    array_path = os.path.join(source_path, 'array')
#    call(['rm', '-rf', array_path])
    shutil.rmtree(array_path,ignore_errors=True)
#    call(['mkdir', '-p', array_path])
    os.makedirs(array_path)

    print('Iterating over all categories: ', categories)
    category_lengths = []
    for category_idx, category in enumerate(categories):
        print('categories:', category)
        category_path = os.path.join(img_path, category)
        img_files = sorted(os.listdir(category_path))
        category_lengths.append(len(img_files))
        for img_idx, img_file in tqdm(enumerate(img_files)):
            this_img_path = os.path.join(category_path, img_file)
            img = load_img(this_img_path, target_size=(img_dim, img_dim))

            img_name = '{}-img-{}-{}'.format(img_idx, category, category_idx)
            label_name = '{}-label-{}-{}'.format(img_idx, category, category_idx)

            label = np.eye(len(categories), dtype = np.float32)[category_idx]

            img_array_path = os.path.join(array_path, img_name)
            img_label_path = os.path.join(array_path, label_name)

            np.save(img_array_path, img)
            np.save(img_label_path, label)
    category_lengths = np.array(category_lengths) / sum(category_lengths)
    category_lengths = list(category_lengths / max(category_lengths))
    category_rounds = {cat: min(int(np.round(1 / l)), 10) for cat, l in zip(categories, category_lengths)}
    return category_rounds
