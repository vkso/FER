import myMethod as myMethod
from datetime import datetime
from customParameters import *
from tensorflow import keras
# show tensor image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import os

train_file_path = "./data/FER2013/train.csv"
test_public_path = "./data/FER2013/public_test.csv"
test_private_path = "./data/FER2013/private_test.csv"

train_data = myMethod.get_dataset_train(train_file_path)
public_test_data = myMethod.get_dataset_test(test_public_path)
private_test_data = myMethod.get_dataset_test(test_private_path)

train_data = train_data.map(myMethod.preprocess_traindata)
public_test_data = public_test_data.map(myMethod.preprocess_testdata)
private_test_data = private_test_data.map(myMethod.preprocess_testdata)

# xxx = next(iter(public_test_data))
# original = xxx[0][1]
# original = tf.reshape(original, [1, 48, 48, 1])
# print(original.shape)
#
# after = tf.image.random_crop(original, size=[1, 42, 42, 1])
# print(after.shape)
# original = tf.reshape(original, [48, 48])
# after = tf.reshape(after, [42, 42])
#
# myMethod.visualize(original, after)
# plt.show()


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------


def tmpvisualize(lu, ru, ld, rd, original):
    fig = plt.figure()
    plt.subplot(3, 2, 1)
    plt.title('Original Image')
    plt.imshow(lu, cmap='gray')

    plt.subplot(3, 2, 2)
    plt.title('Augmented Image')
    plt.imshow(ru, cmap='gray')

    plt.subplot(3, 2, 3)
    plt.title('Augmented Image')
    plt.imshow(ld, cmap='gray')

    plt.subplot(3, 2, 4)
    plt.title('Augmented Image')
    plt.imshow(rd, cmap='gray')

    plt.subplot(3, 2, 5)
    plt.title('Augmented Image')
    plt.imshow(original, cmap='gray')

# tmpvisualize(left_up, right_up, left_down, right_down, original)
# plt.show()
# myMethod.visualize(original, right_down)
# plt.show()

# ------------------------------------------------------------------------------
# TensorBoard
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Save checkpoints
checkpoint_path = "./train_history/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True)
    # period=10)



def singleGPU():
    model = myMethod.create_myVGG()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.fit(train_data, epochs=TOTAL_EPOCHS, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE,
              callbacks=[tensorboard_callback, cp_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE)

def multiGPUs():
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = myMethod.create_myVGG()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])

    model.fit(train_data, epochs=TOTAL_EPOCHS, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE // GPUS,
              callbacks=[tensorboard_callback, cp_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE // GPUS)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE // GPUS)

def trainModel(gpus):
    if gpus == 1:
        singleGPU()
    else:
        multiGPUs()

trainModel(GPUS)
