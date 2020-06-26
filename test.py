import tensorflow as tf
from customParameters import *
import myMethod as myMethod


test_private_path = "./data/FER2013/private_test.csv"
private_test_data = myMethod.get_dataset(test_private_path)
private_test_data = private_test_data.map(myMethod.preprocess_traindata)

model = myMethod.create_myModel()
model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

for i in range(10, 81, 10):
    load_path = './train_history/cp-'+ str(i).zfill(4) + '.ckpt'

    model.load_weights(load_path)
    loss, acc = model.evaluate(private_test_data,
                               steps=TOTAL_TEST // BATCH_SIZE)
    # print("Epoch {}, test loss: {:5.2f}%, accuracy: {:5.2f}%".\
    #       format(str(i).zfill(4), loss*100, acc*100))