from os import listdir
from os.path import isfile, join
from random import random
import pickle
import numpy as np
import keras
import scipy.io
import os
import csv


def create_partition(dataset_path):
    words = ['silence', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
             'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
             'up', 'wow', 'yes', 'zero']
    class_words = ['silence', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
                   'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                   'up', 'wow', 'yes', 'zero']
    partition = {'train': [],
                 'validation': [],
                 'test': []}
    labels = {}

    for word in words:
        train_path = dataset_path + '\\train\\audio\\' + word + '\\'
        train_files = [f for f in listdir(train_path) if (isfile(join(train_path, f)) and f.endswith('.mat'))]
        for f in train_files:
            ID = word + "\\" + f
            if random() < 0.7:
                partition['train'].append(ID)
            elif random() > 0.5:
                partition['validation'].append(ID)
            else:
                partition['test'].append(ID)
            if word in class_words:
                labels[ID] = word
            else:
                labels[ID] = 'unknown'

    pickle.dump(partition, open('partition.p', 'wb'))
    pickle.dump(labels, open('labels.p', 'wb'))


def create_test_partition(dataset_path):
    partition = {'test': []}

    test_path = dataset_path + '\\test\\audio\\'
    test_files = [f for f in listdir(test_path) if (isfile(join(test_path, f)) and f.endswith('.mat'))]
    for ID in test_files:
        partition['test'].append(ID)

    pickle.dump(partition, open('test_partition.p', 'wb'))


def load_partition(partition_fname, labels_fname='none'):
    partition = pickle.load(open(partition_fname, "rb"))
    if labels_fname != 'none':
        labels = pickle.load(open(labels_fname, "rb"))
        return partition, labels
    else:
        return partition


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels={}, cnn_included=False, dataset_path='', batch_size=32, dim=(32, 32, 32),
                 n_channels=1, n_classes=31, test_generator=False, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataset_path = dataset_path
        self.class_words = ['silence', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
                            'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                            'up', 'wow', 'yes', 'zero']
        self.cnn_included = cnn_included
        self.test_generator = test_generator

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if not self.test_generator:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X



    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""

        # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.cnn_included:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim))

        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.test_generator:
                file_path = self.dataset_path + '\\test\\audio\\' + ID
            else:
                file_path = self.dataset_path + '\\train\\audio\\' + ID
            if self.cnn_included:
                X[i, ] = np.expand_dims(scipy.io.loadmat(file_path)['x'], 3)
            else:
                X[i, ] = scipy.io.loadmat(file_path)['x']

            # Store class
            if not self.test_generator:
                y[i] = self.class_words.index(self.labels[ID])

        if self.test_generator:
            return X
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def predict(model_path, data_generator, partition, model=None):
    json_file = open(model_path + '\\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json, custom_objects={'gabor_init': gabor_init(())})

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    loaded_model.load_weights(model_path + '\\weights.h5')
    print('model loaded')

    scores = loaded_model.evaluate_generator(data_generator, verbose=True)
    print(scores)
    print("loss=" + str(scores[0]) + "-acc=" + str(scores[1]) + "\n")

    # predictions = loaded_model.predict_generator(data_generator)
    # print('predictions made')
    # #print(predictions)
    # class_words = ['silence', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house',
    #                'left', 'marvin',
    #                'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
    #                'up', 'wow', 'yes', 'zero']
    # predictions = np.argmax(predictions, axis=1).tolist()
    # results = [class_words[i] for i in predictions]
    #
    # final_prediction_class = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    # for n, i in enumerate(results):
    #     if i not in final_prediction_class:
    #         results[n] = 'unknown'
    #
    # print('class inferences done')
    # csv_data = [['fname', 'label']]
    # partition = partition['test']
    #
    # for i in range(0, len(results)):
    #     csv_data.append([partition[i].replace('mat', 'wav'), results[i]])
    # print('csv data created')
    # with open(model_path + "\\submission.csv", 'w', newline='') as output_file:
    #     writer = csv.writer(output_file)
    #     writer.writerows(csv_data)
    # print('output file created')


def train_model(model, results_summary, training_generator, validation_generator, test_generator,
                epochs, model_file_name):
    try:
        # Create target Directory
        os.mkdir(model_file_name)
        print("Directory ", model_file_name, " Created ")
    except FileExistsError:
        print("Directory ", model_file_name, " already exists")

    # opt = keras.optimizers.SGD(lr=0.01, clipvalue=0.5)
    # opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)
    opt = keras.optimizers.adam(clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                  use_multiprocessing=False, epochs=epochs, verbose=True)

    scores = model.evaluate_generator(test_generator, verbose=True)
    print("loss=" + str(scores[0]) + "-acc=" + str(scores[1]) + "\n")
    results_summary.write("loss=" + str(scores[0]) + "-acc=" + str(scores[1]) + "\n")

    model_json = model.to_json()
    with open(model_file_name + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file_name + "/weights.h5")
    print("Saved model to disk")

    with open(model_file_name + "/history", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # layer = model.get_layer("conv2d_1")
    # w = layer.get_weights()
    # weights = {'w': w}
    # scipy.io.savemat('conv_weigths.mat', weights)


def gabor_init(shape, dtype=None):
    w = scipy.io.loadmat('gabor_weights.mat')['w']
    return w
