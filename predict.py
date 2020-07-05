import tensorflow as tf
from customParameters import *
import myMethod as myMethod
import matplotlib.pyplot as plt
import numpy as np


test_private_path = "./data/FER2013/private_test.csv"
private_test_data = myMethod.get_dataset_test(test_private_path)
private_test_data = private_test_data.map(myMethod.preprocess_DAtestdata)

# get standard result
correct_answer = np.loadtxt(test_private_path, dtype=np.int, delimiter=',',
                            skiprows=1, usecols=(0), encoding='utf-8')
print(type(correct_answer))
print(correct_answer.shape)
print(correct_answer)
# correct_answer = correct_answer.repeat(10)

model = myMethod.create_myVGG()
model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])


# xxx = next(iter(private_test_data))
# original = xxx[0]
# emotion = xxx[1]
# print(original.shape)
# print(emotion.shape)

# for i in range(10, 601, 10):
#     load_path = './train_history/cp-'+ str(i).zfill(4) + '.ckpt'
#     model.load_weights(load_path)
#     x = model.predict(private_test_data,
#                       steps=TOTAL_TEST // BATCH_SIZE)
#     predict_result = np.zeros(shape=(3589, 7))
#     for i in range(0, 3589):
#         sum = np.zeros(shape=(1, 7))
#         for j in range(0, 10):
#             sum += x[10*i+j]
#         predict_result[i] = sum
#     y = np.argmax(predict_result, axis=1)
#     z = y - correct_answer
#     sum = np.sum(z == 0)
#
#     print('sum: {}'.format(sum))
#     print('acc: {}'.format(sum/3589))


# load_path = './train_history/cp-'+ str(i).zfill(4) + '.ckpt'
load_path = '/Users/wyc/Downloads/cp-0560.ckpt'
model.load_weights(load_path)
x = model.predict(private_test_data,
                  steps=TOTAL_TEST // BATCH_SIZE)
predict_result = np.zeros(shape=(3589, 7))
for i in range(0, 3589):
    sum = np.zeros(shape=(1, 7))
    for j in range(0, 10):
        sum += x[10*i+j]
    predict_result[i] = sum
y = np.argmax(predict_result, axis=1)
z = y - correct_answer
sum = np.sum(z == 0)

print('sum: {}'.format(sum))
print('acc: {}'.format(sum/3589))

