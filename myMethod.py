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
import tensorflow_addons as tfa
import random


# parameters: bs->batch_size, lc-> label_column, ne->num_epochs
def get_dataset_train(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        label_name=LABEL_COLUMN,
        num_epochs=NUM_EPOCHS
    )
    return dataset

# Get Test Dataset without shuffle operation.
def get_dataset_test(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        label_name=LABEL_COLUMN,
        num_epochs=NUM_EPOCHS,
        shuffle=False
    )
    return dataset


# image augmentation
# horizontal_fold
def horizontal_fold(original):
    augmented = tf.image.flip_left_right(original)
    return augmented


def central_crop(original):
    augmented = tf.image.central_crop(original, central_fraction=0.75)
    augmented = tf.image.resize(augmented, [48, 48], method='bilinear')
    return augmented


def random_rotated(original):
    random_angle = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5)
    rotated = tfa.image.rotate(original, angles=random_angle, interpolation="BILINEAR")
    rotated = tf.image.central_crop(rotated, central_fraction=0.75)
    rotated = tf.image.resize(rotated, [48, 48], method='bilinear')
    return rotated


def image_augmentation(original, process_type):
    if process_type == 0:
        augmented = horizontal_fold(original)
    elif process_type == 1:
        augmented = central_crop(original)
    else:
        augmented = random_rotated(original)
    return augmented


def preprocess_traindata(feature, labels):
    block = feature['pixels']
    block = tf.strings.split(block)
    block = tf.strings.to_number(block)
    block = tf.cast(block, tf.float32) / 255.0

    tmp_feature = tf.reshape(block[0], [1, 48, 48, 1])
    tmp_label = tf.reshape(tf.one_hot(labels[0], 7), [1, 7])
    method_list = tf.random.uniform([BATCH_SIZE//2], minval=0, maxval=3, dtype=tf.int32)
    for i in range(1, BATCH_SIZE):
        current = tf.reshape(block[i], [1, 48, 48, 1])
        # the first 1/2 part will be processed
        if i < BATCH_SIZE//2:
            operation_type = method_list[i]
            current = image_augmentation(current, operation_type)

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
        layers.Conv2D(64, (7, 7), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(64, (7, 7), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool1_1'),
        # Block 2
        layers.Conv2D(128, (5, 5), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(128, (5, 5), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool2_1'),
        # Block 3
        layers.Conv2D(256, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(256, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv3_2'),
        layers.Conv2D(256, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv3_3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool3_1'),
#         # Block 4
#         layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
#         layers.Conv2D(512, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', padding='same', name='conv4_2'),
#         layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(2, strides=2, padding='same', name='pool4_1'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu', name='fc6'),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu', name='fc7'),
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

