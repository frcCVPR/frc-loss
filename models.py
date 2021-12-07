from functools import partial
import json
import os
import numpy as np
import tensorflow as tf

from dataset import create_images_dataset, add_noise
import settings
from settings import (LOSS_FUNCTION, EPOCHS_NO, LEARNING_RATE, BATCH_SIZE,
                      ADD_NOISE, BATCHES_NUMBER, RAMP_DOWN_PERC, DECAY_STEPS,
                      SAVED_MODEL_LOGDIR, RESTORE_EPOCH, EPOCH_FILEPATTERN, BATCH_FILEPATTERN)
from utils import get_new_model_log_path

from network import autoencoder

class FRCUnetModel(tf.keras.Model):
    def __init__(self, logdir, model_path=None,  *args, **kwargs):
        super(FRCUnetModel, self).__init__(**kwargs)
        self.model = autoencoder(*args, **kwargs)
        if model_path is not None:
            self.model.load_weights(model_path)
        self.radial_masks, self.spatial_freq = self.get_radial_masks()
        if logdir is not None:
            self.writer = tf.summary.create_file_writer(logdir)


    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def scale(image):
        scaled = image - tf.math.reduce_min(image)
        return scaled / tf.math.reduce_max(scaled)

    def create_image_summaries(self, images_data, denoised, step, mode='train'):
        original, noisy1, noisy2 = images_data
        tf.summary.image("original_" + mode, self.scale(original), step=step)
        tf.summary.image("noisy1_" + mode, self.scale(noisy1), step=step)
        tf.summary.image("noisy2_" + mode, self.scale(noisy2), step=step)

        tf.summary.image("denoised_ " + mode, self.scale(denoised), step=step)

    @tf.function
    def compute_loss_mae(self, data):
        original, img1, img2 = data
        denoised = self.call(img1)
        loss = tf.math.reduce_mean(tf.keras.losses.MAE(denoised, img2))

        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, original))
        return denoised, loss, loss_original

    @tf.function
    def compute_loss_mse(self, data):
        original, img1, img2 = data
        denoised = self.call(img1)
        loss = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, img2))

        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, original))
        return denoised, loss, loss_original

    @tf.function
    def compute_loss_frc(self, data):
        original, img1, img2 = data
        denoised1 = self.call(img1)
        denoised2 = self.call(img2)

        loss = -self.fourier_ring_correlation(img2, denoised1, self.radial_masks, self.spatial_freq)
        
        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(self.scale(denoised1), self.scale(original)))
        loss = tf.math.reduce_mean(loss)
        return denoised1, loss, loss_original

    def infer(self, data):
        if LOSS_FUNCTION == 'FRC':
            return self.compute_loss_frc(data)
        elif LOSS_FUNCTION == 'L2':
            return self.compute_loss_mse(data)
        elif LOSS_FUNCTION == 'L1':
            return self.compute_loss_mae(data)

    @tf.function
    def train_step(self, data):
        #data contains original, img1, img2.
        if ADD_NOISE == 'EPOCH':
            data = list(data)
            data[1] = add_noise(data[1])
            data[2] = add_noise(data[2])
        with tf.GradientTape() as tape:
            denoised, loss, loss_original = self.infer(data)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_vars = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(grads_vars)
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        if ADD_NOISE == 'EPOCH':
            data = list(data)
            data[1] = add_noise(data[1])
            data[2] = add_noise(data[2])
        denoised, loss, loss_original = self.infer(data)
        return {'loss': loss}


    @tf.function
    def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 3, 1, 2])
        image2 = tf.transpose(image2, perm=[0, 3, 1, 2])
        image1 = tf.cast(image1, tf.complex64)
        image2 = tf.cast(image2, tf.complex64)
        rn = tf.cast(rn, tf.complex64)
        fft_image1 = tf.signal.fftshift(tf.signal.fft2d(image1), axes=[2, 3])
        fft_image2 = tf.signal.fftshift(tf.signal.fft2d(image2), axes=[2, 3])

        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
        t2 = tf.multiply(fft_image2, rn)
        c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4]))
        c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4])
        c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4])
        frc = tf.math.divide(c1, tf.math.sqrt(tf.math.multiply(c2, c3)))
        frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
        frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc)  # nan

        t = spatial_freq
        y = frc
        riemann_sum = tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[:-1] + y[1:]) / 2.), 0)
        return riemann_sum

    def radial_mask(self, r, cx=128, cy=128, sx=np.arange(0, 256), sy=np.arange(0, 256), delta=1):
        ind = (sx[np.newaxis, :] - cx) ** 2 + (sy[:, np.newaxis] - cy) ** 2
        ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
        ind2 = ind > (r[0] ** 2)
        return ind1 * ind2


    @tf.function
    def get_radial_masks(self):
        freq_nyq = int(np.floor(int(256) / 2.0))
        radii = np.arange(128).reshape(128, 1)  # image size 256, binning = 3
        radial_masks = np.apply_along_axis(self.radial_mask, 1, radii, 128, 128, np.arange(0, 256), np.arange(0, 256), 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        radial_masks = np.expand_dims(radial_masks, 1)

        spatial_freq = radii.astype(np.float32) / freq_nyq
        spatial_freq = spatial_freq / max(spatial_freq)

        return radial_masks, spatial_freq


class Summaries(tf.keras.callbacks.Callback):
    def __init__(self, epoch_restored=-1):
        self.batch_no = 0
        self.epoch_restored = epoch_restored
        self.train_ds, self.val_ds = create_images_dataset()

    def on_epoch_end(self, epoch, logs):
        current_epoch = epoch + self.epoch_restored + 1
        overall_loss_train, overall_loss_val = 0, 0
        overall_loss_original_train, overall_loss_original_val = 0, 0 
        overall_batches_train, overall_batches_val = 0, 0
        idx = 0
        for train_images in self.train_ds.take(100):
            if ADD_NOISE == 'EPOCH':
                train_images = list(train_images)
                train_images[1] = add_noise(train_images[1])
                train_images[2] = add_noise(train_images[2])
            train_denoised, loss_train, loss_original = self.model.infer(train_images)
            overall_loss_train += loss_train
            overall_loss_original_train += loss_original
            overall_batches_train += 1
        for val_images in self.val_ds.take(100):
            if ADD_NOISE == 'EPOCH':
                val_images = list(val_images)
                val_images[1] = add_noise(val_images[1])
                val_images[2] = add_noise(val_images[2])
            val_denoised, loss_val, loss_original = self.model.infer(val_images)
            overall_loss_val += loss_val
            overall_loss_original_val += loss_original
            overall_batches_val += 1

        with self.model.writer.as_default():
            tf.summary.scalar("loss_epoch_train", overall_loss_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_val", overall_loss_val/overall_batches_val, step=current_epoch)
            tf.summary.scalar("loss_epoch_original_train", overall_loss_original_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_original_val", overall_loss_original_val/overall_batches_val,step=current_epoch)

            self.model.create_image_summaries(train_images, train_denoised,
                                              current_epoch, mode='train')
            self.model.create_image_summaries(val_images, val_denoised,
                                              current_epoch, mode='val')


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, logdir, epoch_restored=-1):
        self.logdir = logdir
        self.epoch_restored = epoch_restored
        self.epoch = self.epoch_restored + 1

    def on_epoch_end(self, epoch, logs):
        filename = self.logdir + '/' + EPOCH_FILEPATTERN.format(self.epoch)
        self.model.model.save_weights(filename)
        self.epoch = epoch + 1 + self.epoch_restored + 1

    def on_batch_end(self, batch, logs):
        #the batch arg i the batch in a certain epoch, and we need batch overall
        batch_no = self.epoch * BATCHES_NUMBER + batch
        #check if batch number is a power of two. We will need those model dumps for charts.
        if (batch_no & (batch_no-1) == 0):
            filename = self.logdir + '/' + BATCH_FILEPATTERN.format(batch_no)
            model.model.save_weights(filename)


def exponential_decay(epoch_no):
    return LEARNING_RATE * (1-RAMP_DOWN_PERC) ** (epoch_no/DECAY_STEPS)

class LearningRateSchedulerWithLogs(tf.keras.callbacks.LearningRateScheduler):
    def on_epoch_begin(self, epoch, logs=None):
        super(LearningRateSchedulerWithLogs, self).on_epoch_begin(epoch, logs)
        with self.model.writer.as_default():
            tf.summary.scalar("learning_rate", self.model.optimizer.lr, step=epoch)

if __name__ == "__main__":
    tf.random.set_seed(543)

    if SAVED_MODEL_LOGDIR:
        logdir = SAVED_MODEL_LOGDIR
        model_path = SAVED_MODEL_LOGDIR + '/' + EPOCH_FILEPATTERN.format(RESTORE_EPOCH)
        save_model_callback = SaveModel(logdir, RESTORE_EPOCH)
        summaries_callback = Summaries(RESTORE_EPOCH)
    else:
        _, _, logdir = get_new_model_log_path()
        os.makedirs(logdir)
        with open(os.path.join(logdir, 'params.json'), 'w') as f:
            data = {k: v for k, v in vars(settings).items() if k.isupper()}
            f.write(json.dumps(data))
        model_path = None
        save_model_callback = SaveModel(logdir)
        summaries_callback = Summaries()

    model = FRCUnetModel(logdir, model_path=model_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer)
    train_ds, val_ds = create_images_dataset()
    callbacks = [
        save_model_callback,
        tf.keras.callbacks.TensorBoard(logdir + '/logs', profile_batch='10,20'),
        summaries_callback,
        LearningRateSchedulerWithLogs(exponential_decay)
    ]
    model.fit(train_ds, callbacks=callbacks, epochs=EPOCHS_NO, validation_data=val_ds)


