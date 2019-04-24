from dataset_prep import create_partition, train_model
from sample_models import basic_lstm3, cnn_lstm, basic_lstm1
import pickle
import os
from keras.optimizers import RMSprop
from dataset_prep import create_test_partition, load_partition, DataGenerator, predict

dataset_path = 'C:\\Users\\18ae5\\Desktop\\Datasets\\Tensorflow_ASR_Challenge'
# create_partition(dataset_path)
partition, labels = load_partition('partition.p', 'labels.p')

epochs = 50
params = {'dim': (23, 98),
          'batch_size': 256,
          'n_classes': 31,
          'n_channels': 1,
          'shuffle': True,
          'cnn_included': True,
          'dataset_path': dataset_path}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)


results_summary = open("results.txt", "a")
results_summary.write("Training results:\n")
results_summary.write("CNN: fixed gabor weights without pooling + single layer LSTM:\n")


# model = basic_lstm1(batch_input_shape=(None, 39, 98), batch_normalization=False)
# print(model.summary())
# train_model(model, results_summary, training_generator, validation_generator, test_generator, epochs=epochs,
#             model_file_name="lstm1")
#
#
# model = basic_lstm3(batch_input_shape=(None, 39, 98), batch_normalization=False)
# print(model.summary())
# train_model(model, results_summary, training_generator, validation_generator, test_generator, epochs=epochs,
#             model_file_name="lstm_mfcc_final")

# model = cnn_lstm(init_with_gabor=True, cnn_pooling=True, train_cnn=False)
# print(model.summary())
# train_model(model, results_summary, training_generator, validation_generator, test_generator, epochs=epochs,
#             model_file_name="cnn_lstm_final")
#
# model = cnn_lstm(init_with_gabor=True, cnn_pooling=True, train_cnn=False)
# print(model.summary())
# train_model(model, results_summary, training_generator, validation_generator, test_generator, epochs=epochs,
#             model_file_name="cnn_lstm_1")


# params = {'dim': (23, 98),
#           'batch_size': 6,
#           'n_classes': 31,
#           'n_channels': 1,
#           'shuffle': False,
#           'cnn_included': True,
#           'dataset_path': dataset_path}
#
# # create_test_partition(dataset_path)
# partition = load_partition('test_partition.p')
# test_generator = DataGenerator(partition['test'], test_generator=True, **params)


model_path = 'cnn_lstm_2'
predict(model_path=model_path, data_generator=test_generator, partition=partition)

# model_path = 'cnn_lstm_2'
# predict(model_path=model_path, data_generator=test_generator, partition=partition)

