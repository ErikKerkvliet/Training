import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATA_FOLDER = '/home/erik/PycharmProjects/TrainingData/Data/Images'
CATEGORIES = ['yes', 'no']


class Main:

    def __init__(self):
        self.training_data = []

    def create_training_data(self):
        for category in CATEGORIES:
            path = os.path.join(DATA_FOLDER, category)
            classification_numbers = CATEGORIES.index(category)
            for image in os.listdir(path):
                images = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                self.training_data.append([images, classification_numbers])


main = Main()
main.create_training_data()

random.shuffle(main.training_data)

x = []
y = []

for features, label in main.training_data:
    x.append(features)
    y.append(label)

# ? width height gray_scale
x = np.array(x).reshape(-1, 11, 108, 1)

import pickle

pickle_out = open('x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

import tensorflow as tf

