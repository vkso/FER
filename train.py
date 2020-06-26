import myMethod as myMethod
from datetime import datetime
from customParameters import *
from tensorflow import keras
# show tensor image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

train_file_path = "./data/FER2013/train.csv"
test_public_path = "./data/FER2013/public_test.csv"
test_private_path = "./data/FER2013/private_test.csv"

train_data = myMethod.get_dataset(train_file_path)
public_test_data = myMethod.get_dataset(test_public_path)
private_test_data = myMethod.get_dataset(test_private_path)

train_data = train_data.map(myMethod.preprocess_traindata)
public_test_data = public_test_data.map(myMethod.preprocess_traindata)
private_test_data = private_test_data.map(myMethod.preprocess_traindata)

# xxx = next(iter(public_test_data))
# original = xxx[0][1]
# original = tf.reshape(original, [1, 48, 48, 1])
# print(original.shape)


# ------------------------------------------------------------------------------
# 四个拐角的裁切,左上、右上，左下，右下
# left_up = original[:42, :42, :]
# # original = tf.reshape(original, [48, 48])
# left_up = tf.image.resize(left_up, [48, 48], method='bilinear')
# left_up = tf.reshape(left_up, [48, 48])
#
# right_up = original[:42, 6:, :]
# # original = tf.reshape(original, [48, 48])
# right_up = tf.image.resize(right_up, [48, 48], method='bilinear')
# right_up = tf.reshape(right_up, [48, 48])
#
# left_down = original[6:, :42, :]
# # original = tf.reshape(original, [48, 48])
# left_down = tf.image.resize(left_down, [48, 48], method='bilinear')
# left_down = tf.reshape(left_down, [48, 48])
#
# right_down = original[6:, 6:, :]
# # original = tf.reshape(original, [48, 48])
# right_down = tf.image.resize(right_down, [48, 48], method='bilinear')
# right_down = tf.reshape(right_down, [48, 48])
#
# original = tf.reshape(original, [48, 48])
# ------------------------------------------------------------------------------

# 水平翻转
# horizontal_fold = tf.image.flip_left_right(original)
# original = tf.reshape(original, [48, 48])
# horizontal_fold = tf.reshape(horizontal_fold, [48, 48])

# # 单独中心裁剪
# central_crop = tf.image.central_crop(original, central_fraction=0.75)
# central_crop = tf.image.resize(central_crop, [48, 48], method='bilinear')
# print(central_crop.shape)
# original = tf.reshape(original, [48, 48])
# central_crop = tf.reshape(central_crop, [48, 48])

# # 旋转图像后中心裁剪
# random_angle = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5)
# rotated = tfa.image.rotate(original, angles=random_angle, interpolation="BILINEAR")
# rotated = tf.image.central_crop(rotated, central_fraction=0.75)
# rotated = tf.image.resize(rotated, [48, 48], method='bilinear')
# original = tf.reshape(original, [48, 48])
# rotated = tf.reshape(rotated, [48, 48])
# print(rotated.shape)



# def tmpvisualize(lu, ru, ld, rd, original):
#     fig = plt.figure()
#     plt.subplot(3, 2, 1)
#     plt.title('Original Image')
#     plt.imshow(lu, cmap='gray')
#
#     plt.subplot(3, 2, 2)
#     plt.title('Augmented Image')
#     plt.imshow(ru, cmap='gray')
#
#     plt.subplot(3, 2, 3)
#     plt.title('Augmented Image')
#     plt.imshow(ld, cmap='gray')
#
#     plt.subplot(3, 2, 4)
#     plt.title('Augmented Image')
#     plt.imshow(rd, cmap='gray')
#
#     plt.subplot(3, 2, 5)
#     plt.title('Augmented Image')
#     plt.imshow(original, cmap='gray')
#
#
#
# tmpvisualize(left_up, right_up, left_down, right_down, original)
# plt.show()
# myMethod.visualize(original, horizontal_fold)
# plt.show()

# ------------------------------------------------------------------------------

logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

def singleGPU():
    model = myMethod.create_myModel()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    model.fit(train_data, epochs=90, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE,
              callbacks=[tensorboard_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE)

def multiGPUs():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = myMethod.create_myModel()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])
    model.fit(train_data, epochs=90, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE,
              callbacks=[tensorboard_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE)

def trainModel(gpus):
    if gpus == 1:
        singleGPU()
    else:
        multiGPUs()

trainModel(GPUS)