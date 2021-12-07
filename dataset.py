
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp

from settings import (BATCH_SIZE, BUFFER_SIZE, ADD_NOISE, NOISE_TYPE, CHANNELS,
                      IMIGES_WIDTH, IMIGES_HEIGHT, STDDEV,  IMAGES_PATH_TRAIN,
                      IMAGES_PATH_VAL, IMAGE_TYPE, ODD_SUFFIX, EVEN_SUFFIX, ORIGINAL_SUFFIX)

LOGNORM_DISTR = tfp.distributions.LogNormal(0, 1.3)

def create_list_dataset(data_dir):
    return tf.data.Dataset.list_files(data_dir, shuffle=False)


def decode_img(img, channels=CHANNELS):
    if IMAGE_TYPE == 'JPG':
        img = tf.image.decode_jpeg(img, channels)
    elif IMAGE_TYPE == 'TIFF':
        img = tfio.experimental.image.decode_tiff(img)
    elif IMAGE_TYPE == 'PNG':
        img = tf.image.decode_png(img, channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_image(file_path, channels=CHANNELS):
    img = tf.io.read_file(file_path)
    img = decode_img(img, channels)
    img = img - tf.reduce_min(img)
    if tf.reduce_max(img) !=0:
        img = img/tf.reduce_max(img)
    return img - 0.5


def process_image_single(file_path):
    img = load_image(file_path, channels=3)
    img = tf.image.random_crop(img, [IMIGES_WIDTH, IMIGES_HEIGHT, 3])
    img = tf.image.rgb_to_grayscale(img)
    return img


def process_image_pair(file_path):
    """
    Create dataset from pairs of images already containing noise
    """
    img_odd = load_image(file_path)
    path_even = tf.strings.regex_replace(file_path, ODD_SUFFIX, EVEN_SUFFIX)
    img_even = load_image(path_even)
    path_original = tf.strings.regex_replace(file_path, ODD_SUFFIX, ORIGINAL_SUFFIX)
    img_original = load_image(path_original) 
    return img_original, img_odd, img_even


def process_image_double(file_path):
    """
    Create a datset out of an image adding two instances of noise two it.
    It will be used by dataset map, so the same two instances of noise will be used in every
    epoch, for the same images. It also returns the clean image for tb statistics
    """
    img = process_image_single(file_path)
    img_noise1 = add_noise(img)
    img_noise2 = add_noise(img)
    return img, img_noise1, img_noise2

def process_image_double_clean(file_path):
    """
    Create a dataset yielding two identical images. We will add noise later, during
    the training procedure, so that the noise realisation is different every epoch.
    """
    img1 = process_image_single(file_path)
    img2 = tf.identity(img1)
    return img1, img1, img2

def create_images_one_set(imiges_path, shuffle=True, batch=True, add_noise=ADD_NOISE):
    fnames_dataset = create_list_dataset(imiges_path)
    mapping_functions = {
        'NO': process_image_pair,
        'PERMANENT': process_image_double,
        'EPOCH': process_image_double_clean
    }
    mapping_func = mapping_functions[add_noise]
    dataset = fnames_dataset.map(mapping_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    if batch:
        dataset = dataset.batch(BATCH_SIZE)
    return dataset


def create_images_dataset():
    train_set = create_images_one_set(IMAGES_PATH_TRAIN)
    val_set = create_images_one_set(IMAGES_PATH_VAL, shuffle=False)
    return train_set, val_set


def get_iterator(dataset):
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def get_noise_gaussian(shape, loc, scale):
    return tf.random.normal(shape=shape, mean=loc, stddev=scale)


def get_noise_lognormal(shape, loc, scale):
    lognorm_noise = LOGNORM_DISTR.sample(shape, loc=loc, scale=scale)
    return lognorm_noise - 1


def add_noise(next_elem, loc=0, scale=STDDEV, normalize=False, noise_type=NOISE_TYPE):
    if noise_type == 'GAUSSIAN':
        noise_distribution = get_noise_gaussian(tf.shape(next_elem), loc, scale)
    else:
        noise_distribution = get_noise_lognormal(tf.shape(next_elem), loc, scale)
    noised = next_elem + noise_distribution
    if normalize:
        noised = (noised - np.min(noised)) / np.ptp(noised)
    return noised

if __name__ == '__main__':
    for epoch in range(3):
        print("next epoch")
        train_ds, val_ds = create_images_dataset()
        for images in train_ds.take(3):
            print(tf.math.reduce_min(images[1]), tf.math.reduce_max(images[1]))

