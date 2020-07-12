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
import io
import seaborn as sn


# parameters: bs->batch_size, lc-> label_column, ne->num_epochs
def get_dataset_train(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE_TRAIN,
        label_name=LABEL_COLUMN,
        num_epochs=NUM_EPOCHS
    )
    return dataset

# Get Test Dataset without shuffle operation.
def get_dataset_test(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE_TEST_DA,
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


def random_crop(original):
    augmented = tf.image.random_crop(original, size=[1, 42, 42, 1])
    augmented = tf.image.resize(augmented, [48, 48], method='bilinear')
    return augmented


def random_rotated(original):
    random_angle = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5)
    rotated = tfa.image.rotate(original, angles=random_angle, interpolation="BILINEAR")
    rotated = tf.image.central_crop(rotated, central_fraction=0.75)
    rotated = tf.image.resize(rotated, [48, 48], method='bilinear')
    return rotated


def image_process(original, process_type):
    if process_type == 0:
        augmented = horizontal_fold(original)
    elif process_type == 1:
        augmented = central_crop(original)
    elif process_type == 2:
        augmented = random_crop(original)
    else:
        augmented = random_rotated(original)
    return augmented


def image_augmentation(current, index, method_list):
    if index < BATCH_SIZE_TRAIN // 2:
        operation_type = method_list[index]
        current = image_process(current, operation_type)
    return current


def preprocess_traindata(feature, labels):
    block = feature['pixels']
    block = tf.strings.split(block)
    block = tf.strings.to_number(block)
    block = tf.cast(block, tf.float32) / 255.0

    tmp_feature = tf.reshape(block[0], [1, 48, 48, 1])
    tmp_label = tf.reshape(tf.one_hot(labels[0], 7), [1, 7])
    method_list = tf.random.uniform([BATCH_SIZE_TRAIN//2], minval=0, maxval=4, dtype=tf.int32)
    for i in range(1, BATCH_SIZE_TRAIN):
        current = tf.reshape(block[i], [1, 48, 48, 1])
        current = image_augmentation(current, i, method_list)

        tmp_feature = tf.concat([tmp_feature, current], axis=0)
        current = tf.reshape(tf.one_hot(labels[i], 7), [1, 7])
        tmp_label = tf.concat([tmp_label, current], axis=0)
    feature = tmp_feature
    labels = tmp_label
    return feature, labels

def preprocess_testdata(feature, labels):
    block = feature['pixels']
    block = tf.strings.split(block)
    block = tf.strings.to_number(block)
    block = tf.cast(block, tf.float32) / 255.0

    tmp_feature = tf.reshape(block[0], [1, 48, 48, 1])
    tmp_label = tf.reshape(tf.one_hot(labels[0], 7), [1, 7])

    for i in range(1, BATCH_SIZE_TEST_DA):
        current = tf.reshape(block[i], [1, 48, 48, 1])
        tmp_feature = tf.concat([tmp_feature, current], axis=0)
        current = tf.reshape(tf.one_hot(labels[i], 7), [1, 7])
        tmp_label = tf.concat([tmp_label, current], axis=0)
    feature = tmp_feature
    labels = tmp_label
    return feature, labels


def crop_image_10(feature):
    for i in range(5):
        if i == 0:
            left_up = feature[:, :42, :42, :]
            result = left_up
        elif i == 1:
            right_up = feature[:, :42, 6:, :]
            result = tf.concat([result, right_up], axis=0)
        elif i == 2:
            left_down = feature[:, 6:, :42, :]
            result = tf.concat([result, left_down], axis=0)
        elif i == 3:
            right_down = feature[:, 6:, 6:, :]
            result = tf.concat([result, right_down], axis=0)
        else:
            center = feature[:, 3:45, 3:45, :]
            result = tf.concat([result, center], axis=0)
            result_fold = tf.image.flip_left_right(result)
            result = tf.concat([result, result_fold], axis=0)
            result = tf.image.resize(result, [48, 48], method='bilinear')
    return result


def labels_10(label):
    res = tf.repeat(label, repeats=10, axis=0)
    return res


def preprocess_DAtestdata(feature, labels):
    block = feature['pixels']
    block = tf.strings.split(block)
    block = tf.strings.to_number(block)
    block = tf.cast(block, tf.float32) / 255.0

    tmp_feature = tf.reshape(block[0], [1, 48, 48, 1])
    tmp_feature = crop_image_10(tmp_feature)
    tmp_label = tf.reshape(tf.one_hot(labels[0], 7), [1, 7])
    tmp_label = labels_10(tmp_label)

    for i in range(1, BATCH_SIZE_TEST_DA):
        current = tf.reshape(block[i], [1, 48, 48, 1])
        current = crop_image_10(current)
        tmp_feature = tf.concat([tmp_feature, current], axis=0)

        current = tf.reshape(tf.one_hot(labels[i], 7), [1, 7])
        current = labels_10(current)
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
        keras.Input(shape=(48, 48, 1)),

        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv1_1'),

        layers.MaxPooling2D(2, strides=2, padding='same', name='pool1_1'),
        # Block 2
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      name='conv2_1'),

        layers.MaxPooling2D(2, strides=2, padding='same', name='pool2_1'),
        # Block 3
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      name='conv3_1'),

        layers.MaxPooling2D(2, strides=2, padding='same', name='pool3_1'),
        # --------------------Full conact layer --------------------------------
        layers.Flatten(),
        layers.Dense(1024, activation='relu', name='fc1'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax', name='fc2')
    ])
    return model

def create_myVGG():
    model = keras.models.Sequential([
        keras.Input(shape=(48, 48, 1)),
        # block1
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv1_1'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv1_2'),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool1_1'),

        # block2
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      name='conv2_1'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      name='conv2_2'),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool2_1'),

        # block3
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      name='conv3_1'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      name='conv3_2'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      name='conv3_3'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      name='conv3_4'),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool3_1'),

        # block4
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv4_1'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv4_2'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv4_3'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv4_4'),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool4_1'),

        # block5
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv5_1'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv5_2'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv5_3'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      name='conv5_4'),
        layers.MaxPooling2D(2, strides=2, padding='same', name='pool5_1'),

        layers.AveragePooling2D(pool_size=1, strides=1, name='ap2d'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu', name='fc1'),
        # layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(7, activation='softmax', name='fc2')

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


# plot confusion matrix of predict and groundTruth
def plot_heat_map(predicted, groundTruth):
    cm = tf.math.confusion_matrix(predicted, groundTruth)
    array = cm.numpy()
    np_array = np.around(
        array.astype('int') / array.sum(axis=1)[:, np.newaxis],
        decimals=2)
    ax = sn.heatmap(np_array, annot=True, cmap='YlGnBu',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()

