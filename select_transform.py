import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from PIL import Image
import cv2

CLASS_NAMES = ['anger', 'disgust', 'fear', 'happy', 'sad',
               'surprised', 'normal']

SELECTED_IMAGE = [
    [[110, 139, 560, 676, 678, 807, 1408, 1431, 1886, 1924],
     [2043, 2054, 2100, 2162, 2214, 2217, 2447, 2492, 2645, 3127]],
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


with open('private_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    # dataset = [['emotion', ['255 255 0 255 ...']]]
    dataset = [row for row in reader]


# dataset = [['emotion', nparray[48*48]], [], [], ...]
for i in range(1, 3590):
    dataset[i][1] = dataset[i][1].split()
    for index, val in enumerate(dataset[i][1]):
        dataset[i][1][index] = int(val)
    dataset[i][1] = np.reshape(dataset[i][1], (48, 48))


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




# show_selected_140(dataset)
# show_color_140(dataset)

# save_jpg_file()
# show_random_100(dataset)