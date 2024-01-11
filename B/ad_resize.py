import numpy as np
import tensorflow as tf

def resize_image(train_images_normalized,val_images_normalized,test_images_normalized):
    # resize training dataset
    train_images_resized = tf.image.resize(train_images_normalized, [224, 224]).numpy()
    # resize validation dataset
    val_images_resized = tf.image.resize(val_images_normalized, [224, 224]).numpy()
    # resize testing dataset
    test_images_resized = tf.image.resize(test_images_normalized, [224, 224]).numpy()
    return train_images_resized, val_images_resized, test_images_resized