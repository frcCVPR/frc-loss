import tensorflow as tf
import os

#For distributing the code amongst multiple GPUs
#mirrored_strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 10
LEARNING_RATE = 0.0003
RAMP_DOWN_PERC = 0.3
DECAY_STEPS = 10

#Images dataset location
IMAGES_DIR_TRAIN = ''
IMAGES_PATH_TRAIN = IMAGES_DIR_TRAIN + '/*.jpg'
IMAGES_PATH_VAL = ''
POSSIBLE_IMAGE_TYPES = ['JPG', 'TIFF', 'PNG']
IMAGE_TYPE = 'JPG'

# If ADD_NOISE is set to False it means that we have paired files with noise already added .
# The code will take all the filenames that match IMAGES_PATH_TRAIN and pair them with
# same filename just with changed suffix (ODD_SUFFIX -> EVEN_SUFFIX)
# The code can be found in process_image_pair function in dataset
ADD_NOISE = 'EPOCH'
assert ADD_NOISE in ['NO', 'PERMANENT', 'EPOCH']
NOISE_TYPE = 'GAUSSIAN'
assert NOISE_TYPE in ['GAUSSIAN', 'LOGNORMAL']

ODD_SUFFIX = 'odd'
EVEN_SUFFIX = 'even'
ORIGINAL_SUFFIX = 'even'
assert IMAGE_TYPE in POSSIBLE_IMAGE_TYPES


IMIGES_WIDTH = 256
IMIGES_HEIGHT = 256
CHANNELS = 1

#General settings
BUFFER_SIZE = 1024
if NOISE_TYPE == 'GAUSSIAN':
    STDDEV = 0.2
else:
    STDDEV = 1.3
LOG_PATH = ''
EPOCHS_NO = 500

#Loss function to use
POSSIBLE_LOSSES = ['FRC', 'L2', 'L1']
LOSS_FUNCTION = 'FRC'
SAVED_MODEL_LOGDIR = None
RESTORE_EPOCH = 0
assert LOSS_FUNCTION in POSSIBLE_LOSSES
EPOCH_FILEPATTERN = "saved-model-epoch-{}"
BATCH_FILEPATTERN = "saved-model-batch-{}"

try:
    from local_settings import *
except ImportError:
    pass
BATCHES_NUMBER = int(len([name for name in os.listdir(IMAGES_DIR_TRAIN) if os.path.isfile(IMAGES_DIR_TRAIN + '/' + name)])/BATCH_SIZE)
if ADD_NOISE == 'NO':
    BATCHES_NUMBER=int(BATCHES_NUMBER/2) + 1
