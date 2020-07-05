import tensorflow as tf
from customParameters import *
import myMethod as myMethod
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras import backend as K


test_private_path = "./data/FER2013/private_test.csv"
private_test_data = myMethod.get_dataset_test(test_private_path)
private_test_data = private_test_data.map(myMethod.preprocess_testdata)


model = myMethod.create_myVGG()
model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])


xxx = next(iter(private_test_data))
original = xxx[0][7]
imgshow = tf.reshape(original, [48, 48])
original = tf.reshape(original, [1, 48, 48, 1])
# emotion = xxx[1]
print(original.shape)
# print(emotion.shape)

# Show Image
# --------------------------------------
# plt.imshow(imgshow, cmap='gray')
# plt.show()


# load_path = './train_history/cp-'+ str(i).zfill(4) + '.ckpt'
load_path = '/Users/wyc/Downloads/cp-0560.ckpt'
model.load_weights(load_path)
input_shape = (None, 48, 48, 1)
model.build(input_shape)
model.summary()



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
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

# x = tf.reshape(generate_pattern('conv2_1', 0), [120, 120])
# plt.imshow(x, cmap='gray')
# plt.show()

def show_kernel(layer_name, kernel_nums):
    for i in range(kernel_nums):
        plt.subplot(2, 5, i + 1)
        x = tf.reshape(generate_pattern(layer_name, i), [visual_size, visual_size])
        plt.imshow(x, cmap='gray')
        # plt.show()

# show_kernel(layer_name='conv5_1', kernel_nums=10)
# plt.show()
# ------------------------------------------------------------------------------

for layer_name in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']:
    size = 48
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    results = tf.reshape(results, [419, 419])
    plt.imshow(results)
    plt.show()




# layer_outputs = [layer.output for layer in model.layers[:10]]
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(original)



#
# first_layer_activation = activations[1]
# print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
# plt.show()

# layer_names = []
# for layer in model.layers[:10]:
#     layer_names.append(layer.name)
# print(layer_names)
#
# images_per_row = 16
#
# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = layer_activation.shape[-1]
#
#     size = layer_activation.shape[1]
#
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = layer_activation[0,
#                                              :, :,
#                                              col * images_per_row + row]
#             # Post-process the feature to make it visually palatable
#             channel_image -= channel_image.mean()
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size : (col + 1) * size,
#                          row * size : (row + 1) * size] = channel_image
#
#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
# plt.show()

