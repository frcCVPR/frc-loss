import tensorflow as tf


def double_conv_block_down(initializer, x):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x


def single_conv_block_down(initializer, x):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x


def last_conv_block_down(x, initializer):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def double_conv_block_up(initializer, skip, x, filters_first=96, filters_second=96):
    x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def single_conv_block_up(initializer, skip, x):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                                   kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def autoencoder():
    inputs = tf.keras.layers.Input(shape=[None, None, 1])
    initializer = tf.keras.initializers.HeNormal()
    skips = [inputs]


    #down sample:                                   input: 256x256
    x = double_conv_block_down(initializer, inputs) #128x128
    skips.append(x)
    for _ in range(3):
        x = single_conv_block_down(initializer, x)  #64x64 -> 32x32 ->16x16
        skips.append(x)                             # 8x8
    x = last_conv_block_down(x, initializer)
    for _ in range(4):
        x = double_conv_block_up(initializer, skips.pop(), x)
    x = double_conv_block_up(initializer, skips.pop(), x, 64, 32)
    x = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    x = tf.random.normal([1, 256, 256, 1])
    a = autoencoder()
    print(a(x))