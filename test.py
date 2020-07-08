import tensorflow as tf
from customParameters import *
import myMethod as myMethod
import argparse

parser = argparse.ArgumentParser(description='For Test Model')
# parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--model', type=str, default='myModel')

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

for i in range(10, 301, 10):
    load_path = './train_history/cp-'+ str(i).zfill(4) + '.ckpt'

    model.load_weights(load_path)
    loss, acc = model.evaluate(private_test_data,
                               steps=TOTAL_TEST // BATCH_SIZE_TEST_DA)

    print("Epoch {}, test loss: {:5.2f}%, accuracy: {:5.2f}%".\
          format(str(i).zfill(4), loss*100, acc*100))