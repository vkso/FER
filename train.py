import myMethod as myMethod
from datetime import datetime
from customParameters import *
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse

# python train.py --gpus 1 --model myModel --train_name fc1024
parser = argparse.ArgumentParser(description='train args')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--model', type=str, default='myModel')
parser.add_argument('--train_name', type=str, default='newTrain')

args = parser.parse_args()


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
checkpoint_path = "./train_history/" + args.train_name + '/' + "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    # save_best_only=True)
    period=10)



def singleGPU():
    if args.model == 'myVGG':
        model = myMethod.create_myVGG()
    else:
        model = myMethod.create_myModel()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    model.summary()

    model.fit(train_data, epochs=TOTAL_EPOCHS, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE_TRAIN,
              callbacks=[tensorboard_callback, cp_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE_TRAIN)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE_TRAIN)

def multiGPUs():
    GPUS = args.gpus
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        if args.model == 'myVGG':
            model = myMethod.create_myVGG()
        else:
            model = myMethod.create_myModel()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])

    model.fit(train_data, epochs=TOTAL_EPOCHS, steps_per_epoch=TOTAL_TRAIN // BATCH_SIZE_TRAIN // GPUS,
              callbacks=[tensorboard_callback, cp_callback],
              validation_data=public_test_data,
              validation_steps=TOTAL_TEST // BATCH_SIZE_TRAIN // GPUS)
    model.evaluate(private_test_data, steps=TOTAL_TEST // BATCH_SIZE_TRAIN // GPUS)

def trainModel():
    if args.gpus == 1:
        singleGPU()
    else:
        multiGPUs()

if __name__ == '__main__':
    trainModel()