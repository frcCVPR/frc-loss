from settings import LOG_PATH

import os

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Dense,
                                     PReLU, BatchNormalization, UpSampling2D, Add)

def filter_variables(key):
    return [x for x in tf.trainable_variables() if key in x.name]


def get_new_model_log_path(suffix='models'):
    models_path = os.path.join(LOG_PATH, suffix)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    sorted_files = sorted(int(f) for f in os.listdir(models_path) if 'json' not in f if '.DS_Store' not in f and 'overall' not in f)
    if not sorted_files:
        new_filename = '1'
    else:
        new_filename = str(int(sorted_files[-1]) + 1)
    return new_filename, models_path, os.path.join(models_path, new_filename)


def stringify_layer(layer):
    layer_string = str(type(layer))
    return layer_string.split('.')[-1].split("'")[0]


def layer_to_string(layer):
    layer_type = stringify_layer(layer)
    additional_data = ''
    if type(layer) is Conv2D:
        additional_data = str(layer.kernel.shape.as_list()) + str(layer.strides)
    if type(layer) is MaxPooling2D:
        additional_data = str(list(layer.pool_size))
    if type(layer) is Dropout:
        additional_data = str(layer.rate)
    if type(layer) is Dense:
        additional_data = str(layer.units)
    if type(layer) is BatchNormalization:
        additional_data = str(layer.momentum)
    return layer_type + ' ' + additional_data


def get_model_as_string(model):
    return "\n".join([layer_to_string(x) for x in model.layers])
