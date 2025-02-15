from tensorflow import keras
from keras import layers 
from keras import optimizers
from keras import callbacks 
# from tensorflow.python.keras.layers import *
# from tensorflow.python import keras
# from tensorflow.python.layers.normalization import BatchNormalization
# from tensorflow.python.keras.optimizers import *
# from  tensorflow.python.keras.optimizers import adam_v2 as Adam
# from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.python.keras import backend as keras
from data import *
from attention import cbam_block, eca_block, se_block
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras.backend as K

attention = [se_block, cbam_block, eca_block]

def binary_focal_loss(gamma=2, alpha=0.75):

    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def identity_block(inp, filters, kernel_size):
    f1, f2, f3 = filters

    x = layers.Conv2D(filters=f1, kernel_size=1, padding='same', kernel_initializer='he_normal')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    add = layers.Add()([inp, x])
    x = layers.Activation('relu')(add)

    return x


def convolutional_block(inp, filters, kernel_size, strides=2):
    f1, f2, f3 = filters

    y = layers.Conv2D(filters=f1, kernel_size=1, padding='same', strides=strides, kernel_initializer='he_normal')(inp)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal')(y)
    y = layers.BatchNormalization()(y)

    shortcut = layers.Conv2D(filters=f3, kernel_size=1, strides=strides, kernel_initializer='he_normal')(inp)
    shortcut = layers.BatchNormalization()(shortcut)

    add = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(add)

    return y


def U_CARFnet(pretrained_weights=None, input_size=(256, 256, 1), phi=3):
    inputs = layers.Input(input_size)

    conv1_channel = 64
    conv1 = layers.Conv2D(conv1_channel, 3, padding='same', kernel_initializer='he_normal')(inputs)  # keras.layers.convolutional
    bn1 = layers.BatchNormalization()(conv1)
    outp1 = layers.Activation('relu')(bn1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(outp1)
    cblock1 = convolutional_block(outp1, [64, 64, 128] , 3)
    if phi >= 1 and phi <= 3:
        cblock1 = attention[phi - 1](cblock1, name='cblock1')
    iblock1 = identity_block(cblock1, (64, 64, 128), 3)
    outp2 = iblock1

    cblock2 = convolutional_block(outp2, (128, 128, 256), 3)
    if phi >= 1 and phi <= 3:
        cblock2 = attention[phi - 1](cblock2, name='cblock2')
    iblock2 = identity_block(cblock2, (128, 128, 256), 3)
    outp3 = iblock2

    cblock3 = convolutional_block(outp3, (256, 256, 512), 3)
    if phi >= 1 and phi <= 3:
        cblock3 = attention[phi - 1](cblock3, name='cblock3')
    iblock3 = identity_block(cblock3, (256, 256, 512), 3)
    outp4 = iblock3

    cblock4 = convolutional_block(outp4, (512, 512, 1024), 3)
    if phi >= 1 and phi <= 3:
        cblock4 = attention[phi - 1](cblock4, name='cblock4')
    iblock4 = identity_block(cblock4, (512, 512, 1024), 3)
    outp5 = iblock4

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(outp5))
    merge6 = layers.concatenate([outp4, up6], axis=3)
    cblock5 = convolutional_block(merge6, (1024, 512, 512), 3, 1)
    #if phi >= 1 and phi <= 3:
       #cblock5 = attention[phi - 1](cblock5, name='cblock5')
    iblock5 = identity_block(cblock5, (1024, 512, 512), 3)

    outp6 = iblock5

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(outp6))
    merge7 = layers.concatenate([outp3, up7], axis=3)
    cblock6 = convolutional_block(merge7, (512, 256, 256), 3, 1)
    #if phi >= 1 and phi <= 3:
       #cblock6 = attention[phi - 1](cblock6, name='cblock6')
    iblock6 = identity_block(cblock6, (512, 256, 256), 3)
    outp7 = iblock6

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(outp7))
    merge8 = layers.concatenate([outp2, up8], axis=3)
    cblock7 = convolutional_block(merge8, (256, 128, 128), 3, 1)
    #if phi >= 1 and phi <= 3:
       #cblock7 = attention[phi - 1](cblock7, name='cblock7')
    iblock7 = identity_block(cblock7, (256, 128, 128), 3)
    outp8 = iblock7

    upl1 = layers.Conv2D(64, 1, activation='relu', padding='same', kernel_initializer="he_normal")(outp8)
    upr1 = layers.Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(upl1)
    )

    upl2 = layers.Conv2D(64, 1, activation='relu', padding='same', kernel_initializer="he_normal")(outp7)
    upr2 = layers.Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(4, 4))(upl2)
    )

    upl3 = layers.Conv2D(64, 1, activation='relu', padding='same', kernel_initializer="he_normal")(outp6)
    upr3 = layers.Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(8, 8))(upl3)
    )

    upl4 = layers.Conv2D(64, 1, activation='relu', padding='same', kernel_initializer="he_normal")(outp5)
    upr4 = layers.Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(16, 16))(upl4)
    )

    merge_final = layers.concatenate([upr1, upr2, upr3, upr4], axis=3)

    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_final)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.Model(inputs, conv10)
    #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall', 'Precision'])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=[binary_focal_loss(alpha=.75, gamma=2)], metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def U_CARFnetrain(batch=1, img_num=30, epochs=150, load_model=None):
    myGene = trainGenerator(batch, "./data1/train", "image", "label", image_num=img_num)
    model = U_CARFnet(load_model)
    model_checkpoint = callbacks.ModelCheckpoint('U_CARFnet.keras', monitor='loss', verbose=1, save_best_only=True)
    log_dir = "logs/fit"
    tensorboard_callback = callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)
    history = model.fit(myGene, steps_per_epoch= int(img_num / batch), epochs=epochs, callbacks=[model_checkpoint, tensorboard_callback])
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend(loc='lower right')
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    return model

if __name__ == '__main__':
    U_CARFnetrain()
