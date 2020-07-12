import tensorflow as tf
from customParameters import *
import myMethod as myMethod
import argparse

# python test.py --model myModel --test_name fc1024 --total_epoch 601
parser = argparse.ArgumentParser(description='For Test Model')
# parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--model', type=str, default='myModel')
parser.add_argument('--test_name', type=str, default='newTest')
parser.add_argument('--total_epoch', type=int, default=601)

args = parser.parse_args()

thisModel = args.model

test_private_path = "./data/FER2013/private_test.csv"
private_test_data = myMethod.get_dataset_test(test_private_path)
private_test_data = private_test_data.map(myMethod.preprocess_testdata)

if thisModel == 'myVGG':
    model = myMethod.create_myVGG()
else:
    model = myMethod.create_myModel()

model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

best_acc = 0
best_epoch = 0
for i in range(10, args.total_epoch, 10):
    load_path = './train_history/' + args.test_name + '/cp-'+ str(i).zfill(4) + '.ckpt'

    model.load_weights(load_path)
    loss, acc = model.evaluate(private_test_data,
                               steps=TOTAL_TEST // BATCH_SIZE_TEST_DA)

    print("Epoch {}, test loss: {:5.2f}%, accuracy: {:5.2f}%".\
          format(str(i).zfill(4), loss*100, acc*100))
    if acc > best_acc:
        best_acc = acc
        best_epoch = i
print('best epoch: {}, acc: {}'.format(best_epoch, best_acc * 100))