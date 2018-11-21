import os
import glob
import numpy as np
import cv2
from random import shuffle

class data_loader:

    def __init__(self, parameters):
        self.data_dir = parameters['data_dir']
        self.data_ext = parameters['image_ext']
        self.batch_size = parameters['batch_size']
        self.noise_len = parameters['noise_length']
        self.dirs = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        self.images = []
        self.pointer = 0

    def load_data(self):

        for dir in self.dirs:
            path = glob.glob(str(self.data_dir + '/' + dir + '/*' + self.data_ext))
            for im_path in path:
                image = cv2.imread(im_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (64, 64))
                self.images.append(image)
        print("No of images loaded:{}".format(len(self.images)))

    def shuffle_data(self):
        shuffle(self.images)
        self.pointer = 0

    def next_batch(self):

        real_images = self.images[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size
        real_labels = np.ones(self.batch_size) - np.random.random_sample(self.batch_size) * 0.2
        gen_labels = np.random.random_sample(self.batch_size) * 0.2
        dis_labels = np.expand_dims(np.append(real_labels, gen_labels), axis=1)

        return real_images, dis_labels

    def noise(self, intg=1):
        noise = np.random.normal(0, 1, size=(self.batch_size * intg, self.noise_len))
        fake_labels = np.expand_dims(np.ones(self.batch_size * intg) - np.random.random_sample(self.batch_size * intg) * 0.2, axis=1)
        return noise, fake_labels
