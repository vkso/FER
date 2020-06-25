#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from customParameters import *
import functools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import layers
from tensorflow import keras

# parameters: bs->batch_size, lc-> label_column, ne->num_epochs
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        label_name=LABEL_COLUMN,
        num_epochs=NUM_EPOCHS
    )
    return dataset

# image augmentation
# horizontal_fold
def horizontal_fold(original):
    augmented = tf.image.flip_left_right(original)
    return augmented


# parameters: bs -> batch_size
def preprocess_traindata(feature, labels):
    block = feature['pixels']
    block = tf.strings.split(block)
    block = tf.strings.to_number(block)
    block = tf.cast(block, tf.float32) / 255.0

    tmp_feature = tf.reshape(block[0], [1, 48, 48, 1])
    tmp_label = tf.reshape(tf.one_hot(labels[0], 7), [1, 7])
    for i in range(1, BATCH_SIZE):
        current = tf.reshape(block[i], [1, 48, 48, 1])
        if i < BATCH_SIZE//2:
            current = horizontal_fold(current)
        tmp_feature = tf.concat([tmp_feature, current], axis=0)
        current = tf.reshape(tf.one_hot(labels[i], 7), [1, 7])
        tmp_label = tf.concat([tmp_label, current], axis=0)
    feature = tmp_feature
    labels = tmp_label
    return feature, labels


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{}:\n {}".format(key, value.numpy()))


def create_myModel():
    model = keras.models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool1_1'),
        # Block 2
        layers.Conv2D(128, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool2_1'),
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(256, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv3_2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool3_1'),
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.Conv2D(512, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv4_2'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool4_1'),

        layers.Flatten(),
        layers.Dense(4096, activation='relu', name='fc6'),
        layers.Dense(4096, activation='relu', name='fc7'),
        layers.BatchNormalization(),
        layers.Dense(7, activation='softmax', name='fc8')
    ])
    return model


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Augmented Image')
    plt.imshow(augmented, cmap='gray')

