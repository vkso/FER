import tensorflow as tf
from customParameters import *
import myMethod as myMethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import argparse

# use method:
# python predict.py --model myModel --type whole_history_epoch
# python predict.py --model myVGG --type whole --load_path /Users/wyc/Downloads/cp-0560.ckpt
# python predict.py --model myModel --type whole_history_epoch --train_name withoutFirstBN --total_epoch 171


parser = argparse.ArgumentParser(description='predicted with confusion matrix')
parser.add_argument('--model', type=str, default='myModel')
parser.add_argument('--type', type=str, default='whole')
parser.add_argument('--load_path', type=str)
parser.add_argument('--train_name', type=str, default='newTrain')
parser.add_argument('--total_epoch', type=int, default=600)
# parser.add_argument('--gpus', type=int, default=1)

args = parser.parse_args()
max_epoch = args.total_epoch

test_private_path = "./data/FER2013/private_test.csv"
private_test_data = myMethod.get_dataset_test(test_private_path)
private_test_data = private_test_data.map(myMethod.preprocess_DAtestdata)

# get standard result
correct_answer = np.loadtxt(test_private_path, dtype=np.int, delimiter=',',
                            skiprows=1, usecols=(0), encoding='utf-8')

# correct_answer = correct_answer.repeat(10)

if args.model == 'myVGG':
    model = myMethod.create_myVGG()
else:
    model = myMethod.create_myModel()

model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])



def get_acc_predict(load_path):
    model.load_weights(load_path)
    x = model.predict(private_test_data,
                      steps=TOTAL_TEST // BATCH_SIZE_TEST_DA)
    predict_result = np.zeros(shape=(3589, 7))
    for i in range(0, 3589):
        sum = np.zeros(shape=(1, 7))
        for j in range(0, 10):
            sum += x[10 * i + j]
        predict_result[i] = sum
    y = np.argmax(predict_result, axis=1)
    z = y - correct_answer
    sum = np.sum(z == 0)

    print('sum: {}'.format(sum))
    print('acc: {}'.format(sum / 3589))
    return y


def get_history_acc(testname):
    max_acc = 0
    best_epoch = 0
    for epoch in range(10, max_epoch, 10):
        load_path = "./train_history/" + testname + '/cp-' + str(epoch).zfill(4) + '.ckpt'
        model.load_weights(load_path)
        x = model.predict(private_test_data,
                          steps=TOTAL_TEST // BATCH_SIZE_TEST_DA)
        predict_result = np.zeros(shape=(3589, 7))
        for i in range(0, 3589):
            sum = np.zeros(shape=(1, 7))
            for j in range(0, 10):
                sum += x[10 * i + j]
            predict_result[i] = sum
        y = np.argmax(predict_result, axis=1)
        z = y - correct_answer
        sum = np.sum(z == 0)

        if sum / 3589 > max_acc:
            max_acc = sum / 3589
            best_epoch = epoch

        print('epoch: {}, correct num: {}, acc: {}'.format(epoch,sum, sum/3589))
    print('epoch: {} -> max acc: {}'.format(best_epoch, max_acc))

if args.type == 'whole':
    load_path = args.load_path
    y = get_acc_predict(load_path)
    myMethod.plot_heat_map(y, correct_answer)


if args.type == 'whole_history_epoch':
    testname = args.train_name
    get_history_acc(testname)
