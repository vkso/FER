import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from PIL import Image
import cv2
import myMethod as myMethod
import argparse
from keras import backend as K
import tensorflow as tf

# python select_transform.py --model myVGG --load_path /Users/wyc/Downloads/cp-0560.ckpt

parser = argparse.ArgumentParser(description='Experiment on selected 140 pics')
parser.add_argument('--model', type=str, default='myVGG')
parser.add_argument('--load_path', type=str)

args = parser.parse_args()


CLASS_NAMES = ['anger', 'disgust', 'fear', 'happy', 'sad',
               'surprised', 'normal']

SELECTED_IMAGE = [
    [[110, 139, 560, 676, 678, 807, 1408, 1431, 1886, 1924],
     [2034, 2054, 2100, 2162, 2214, 2217, 2447, 2492, 2645, 3127]],
    [[1085, 1089, 1245, 1363, 1372, 1486, 1549, 1697, 2717, 3282],
     [383, 419, 614, 986, 1724, 1770, 2221, 2295, 2650, 3093]],
    [[240, 290, 391, 463, 693, 1250, 1284, 1285, 1399, 1506],
     [265, 271, 642, 702, 938, 1112, 1156, 1175, 1310, 3059]],
    [[7, 11, 49, 66, 104, 116, 118, 173, 199, 346],
     [29, 48, 52, 115, 190, 272, 443, 564, 732, 2271]],
    [[143, 285, 318, 328, 356, 520, 545, 721, 1219, 1251],
     [3088, 3162, 3192, 3202, 3296, 3301, 3408, 3514, 3561, 3575]],
    [[17, 36, 122, 134, 161, 196, 382, 580, 723, 2413],
     [2606, 2797, 3147, 3215, 3232, 3272, 3389, 3520, 3544, 3577]],
    [[28, 31, 47, 58, 572, 592, 805, 937, 1469, 1496],
     [57, 62, 283, 399, 441, 798, 1265, 1361, 1515, 3118]]
]

with open('./data/FER2013/private_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    # dataset = [['emotion', ['255 255 0 255 ...']]]
    dataset = [row for row in reader]

# dataset = [['emotion', nparray[48*48]], [], [], ...]
for i in range(1, 3590):
    dataset[i][1] = dataset[i][1].split()
    for index, val in enumerate(dataset[i][1]):
        dataset[i][1][index] = int(val)
    dataset[i][1] = np.reshape(dataset[i][1], (48, 48))



def create_true_label():
    res = []
    label = 0
    for i in range(0, 7):
        for j in range(0, 10):
            res.append(label)
        label += 1
    return np.array(res)


def create_model():
    if args.model == 'myVGG':
        model = myMethod.create_myVGG()
    else:
        model = myMethod.create_myModel()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    load_path = args.load_path
    model.load_weights(load_path)
    model.summary()
    return model


def save_jpg_file():
    for i in range(1, 3590):
        im = Image.fromarray(dataset[i][1].astype(np.uint8))
        ei = int(dataset[i][0])
        path = './testJPG/' + str(i-1) + '_' + CLASS_NAMES[ei] + '.jpg'
        im.save(path)


def show_random_100(dataset):
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        # tmp = rows[i][:35, :35]
        # tmp = rows[i][13:, 13:]
        index = random.randint(1, 3590)
        plt.imshow(dataset[index][1], cmap='gray')
    plt.show()


def show_selected_140(dataset):
    imgIndex = 1
    for i in SELECTED_IMAGE:
        for j in i:
            for item in j:
                plt.subplot(14, 10, imgIndex)
                plt.axis('off')
                plt.imshow(dataset[item+1][1], cmap='gray')
                imgIndex += 1
    plt.show()


def show_color_140(dataset):
    imgIndex = 1
    for i in SELECTED_IMAGE:
        for j in i:
            for item in j:
                plt.subplot(14, 10, imgIndex)
                plt.axis('off')
                heatmap = dataset[item+1][1]
                heatmap = (heatmap).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                plt.imshow(heatmap)
                # plt.imshow(dataset[item+1][1], cmap='gray')
                imgIndex += 1
    plt.show()


def get_standard_70(dataset):
    imgIndex = 0
    res = np.zeros(shape=(70, 48, 48, 1))
    for i in SELECTED_IMAGE:
        for j in i[0]:
            img = np.reshape(dataset[j+1][1], (1, 48, 48, 1))
            res[imgIndex] = img
            imgIndex += 1
    res /= 255.
    return res


def get_ambiguous_70(dataset):
    imgIndex = 0
    res = np.zeros(shape=(70, 48, 48, 1))
    for i in SELECTED_IMAGE:
        for j in i[1]:
            img = np.reshape(dataset[j+1][1], (1, 48, 48, 1))
            res[imgIndex] = img
            imgIndex += 1
    res /= 255.
    return res


def cal_acc_70(selected_data, datatype):
    predict_result = model.predict(selected_data)
    predict_index = np.argmax(predict_result, axis=1)
    true_index = create_true_label()

    sub = predict_index - true_index
    sum = np.sum(sub == 0)
    print('{} - sum: {}'.format(datatype,sum))
    print('{} - acc: {}'.format(datatype, sum / 70))


def show_heat_map(dataset):
    standard = np.reshape(dataset, (70, 1, 48, 48, 1))
    for x in standard:
        # x = tf.convert_to_tensor(x)
        out = model.predict(x)
        print('probability: {}'.format(out))
        predict_index = np.argmax(out)
        print('index: {}, emotion: {}'.format(predict_index,
                                              CLASS_NAMES[predict_index]))
        out = model.output[:, predict_index]

        last_conv_layer = model.get_layer('conv1_1')
        grads = K.gradients(out, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input],
                             [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(64):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        # heatmap /= np.max(heatmap)
        plt.matshow(heatmap)
        plt.show()


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


visual_size = 60
def generate_pattern(layer_name, filter_index, size=visual_size):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    np.random.seed(1314)
    # input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.
    input_img_data = np.random.random((1, size, size, 1))

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(50):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def show_kernel(layer_name, kernel_nums):
    for i in range(kernel_nums):
        plt.subplot(4, 16, i + 1)
        x = tf.reshape(generate_pattern(layer_name, i), [visual_size, visual_size])
        plt.imshow(x, cmap='gray')
    plt.show()


def show_kernel_multilayer(size, layer_list):
    margin = 5
    for layer_name in layer_list:
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))

        for i in range(8):  # iterate over the rows of our results grid
            for j in range(8):  # iterate over the columns of our results grid
                # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                filter_img = generate_pattern(layer_name, i + (j * 8),
                                              size=size)

                # Put the result in the square `(i, j)` of the results grid
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img

        # Display the results grid
        plt.figure(figsize=(20, 20))
        results = tf.reshape(results, [419, 419])
        plt.title(layer_name)
        plt.axis('off')
        plt.imshow(results, cmap='gray')
        plt.show()


def show_conv_result(total_layer_number, image):
    layer_outputs = [layer.output for layer in model.layers[:total_layer_number]]
    activation_model = tf.keras.models.Model(inputs=model.input,
                                             outputs=layer_outputs)
    activations = activation_model.predict(image)

    layer_names = []
    for layer in model.layers[:total_layer_number]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        if not layer_name.startswith('conv'):
            continue
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()




# show_selected_140(dataset)
# show_color_140(dataset)

# save_jpg_file()
# show_random_100(dataset)
# ------------------------------------------------------------------------------
model = create_model()

standard = get_standard_70(dataset)
ambiguous = get_ambiguous_70(dataset)

# cal_acc_70(standard, 'standard')
# cal_acc_70(ambiguous, 'ambigous')

# show_heat_map(ambiguous)
# show_heat_map(standard)

# myVGG
# show_kernel(layer_name='conv1_1', kernel_nums=64)
# layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']

# myModel
# layers = ['conv1_1', 'conv2_1', 'conv3_1']
# show_kernel_multilayer(size=48, layer_list=layers)


standard = np.reshape(standard, (70, 1, 48, 48, 1))
# show_conv_result(33, standard[0])

show_conv_result(7, standard[0])