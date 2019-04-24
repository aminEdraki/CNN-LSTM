import matplotlib.pyplot as plt
import pickle
import numpy as np

import scipy.io

history = pickle.load(open("C:\\Users\\18ae5\\Desktop\\DSP_Project\\Codes&Simulation\\python\\cnn_lstm_final/history", "rb"))
# print(history)
# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

scipy.io.savemat('C:\\Users\\18ae5\\Desktop\\DSP_Project\\Codes&Simulation\\python\\cnn_lstm_final/history.mat', mdict=history)
