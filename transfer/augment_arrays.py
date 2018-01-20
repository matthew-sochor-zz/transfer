import os
import shutil
from subprocess import call

from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def gen_arrays_from_dir(array_dir):
    array_files = sorted(os.listdir(array_dir))
    array_names = list(filter(lambda x: r'-img-' in x, array_files))
    label_names = list(filter(lambda x: r'-label-' in x, array_files))

    assert len(array_names) == len(label_names)

    for array_name, label_name in zip(array_names, label_names):
        array = np.load(os.path.join(array_dir, array_name))
        label = np.load(os.path.join(array_dir, label_name))
        yield array, label, label_name


def gen_augment_arrays(array, label, augmentations, rounds = 1):
    if augmentations is None:
        yield array, label
    else:

        auggen = ImageDataGenerator(featurewise_center = augmentations['featurewise_center'],
                                    samplewise_center = augmentations['samplewise_center'],
                                    featurewise_std_normalization = augmentations['featurewise_std_normalization'],
                                    samplewise_std_normalization = augmentations['samplewise_std_normalization'],
                                    zca_whitening = augmentations['zca_whitening'],
                                    rotation_range = augmentations['rotation_range'],
                                    width_shift_range = augmentations['width_shift_range'],
                                    height_shift_range = augmentations['height_shift_range'],
                                    shear_range = augmentations['shear_range'],
                                    zoom_range = augmentations['zoom_range'],
                                    channel_shift_range = augmentations['channel_shift_range'],
                                    fill_mode = augmentations['fill_mode'],
                                    cval = augmentations['cval'],
                                    horizontal_flip = augmentations['horizontal_flip'],
                                    vertical_flip = augmentations['vertical_flip'],
                                    rescale = augmentations['rescale'])

        array_augs, label_augs = next(auggen.flow(np.tile(array[np.newaxis],
                                                (rounds * augmentations['rounds'], 1, 1, 1)),
                                        np.tile(label[np.newaxis],
                                                (rounds * augmentations['rounds'], 1)),
                                        batch_size=rounds * augmentations['rounds']))

        for array_aug, label_aug in zip(array_augs, label_augs):
            yield array_aug, label_aug


def augment_arrays(project):

    array_path = os.path.join(project['path'], 'array')
    augmented_path = os.path.join(project['path'], 'augmented')
#    call(['rm', '-rf', augmented_path])
    shutil.rmtree(augmented_path,ignore_errors=True)
#    call(['mkdir', '-p', augmented_path])
    os.makedirs(augmented_path)

    if project['augmentations'] is None:
        print('No augmentations selected: copying train arrays as is.')
        files = os.listdir(array_path)
        for file in tqdm(files):
#            call(['cp', os.path.join(array_path, file), augmented_path])
            shutil.copy(os.path.join(array_path, file),augmented_path)

    else:
        print('Generating image augmentations:')

        for img_idx, (array, label, label_name) in tqdm(enumerate(gen_arrays_from_dir(array_path))):
            split_label_name = '-'.join(label_name.split('-')[2:-1])
            for aug_idx, (array_aug, label_aug) in enumerate(gen_augment_arrays(array, label, project['augmentations'], project['category_rounds'][split_label_name])):
                cat_idx = np.argmax(label_aug)
                cat = project['categories'][cat_idx]
                img_name = '{}-{:02d}-img-{}-{}'.format(img_idx, aug_idx,
                                                            cat, cat_idx)
                label_name = '{}-{:02d}-label-{}-{}'.format(img_idx, aug_idx,
                                                            cat, cat_idx)
                aug_path = os.path.join(augmented_path, img_name)
                label_path = os.path.join(augmented_path, label_name)
                np.save(aug_path, array_aug)
                np.save(label_path, label_aug)

    project['is_augmented'] = True
    return project
