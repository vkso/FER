LABEL_COLUMN = 'emotion'
# LABELS = [0, 1, 2, 3, 4, 5, 6]
CATEGORIES = {'Usage' : ['Training', 'PublicTest', 'PrivateTest']}
EMOTION_NAMES = {0 : 'anger', 1 : 'disgust', 2 : 'fear',
                 3 : 'happy', 4 : 'sad', 5 : 'surprised', 6 : 'normal'}
CLASS_NAMES = ['anger', 'disgust', 'fear', 'happy', 'sad',
               'surprised', 'normal']

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST_DA = 37
TOTAL_TRAIN = 28790
TOTAL_TEST = 3589
NUM_EPOCHS = None
TOTAL_EPOCHS = 600