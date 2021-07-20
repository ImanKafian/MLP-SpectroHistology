'''
About: Python script to define four MLP architectures for the classification problem.
Author: Iman Kafian-Attari
Date: 20.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. You do not need to interact with this script directly.
=========================================================
Notes:
1. This script is called from the main script.
2. It includes various MLP models for the classification problem.
3. You can modify the existing MLP models or create new models, if you desire.
4. If you created new models, do not forget to call them from the main script.
=========================================================
TODO for version O.2
1. Add new neural network models.
=========================================================
'''

print(__doc__)

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from tensorflow.keras.losses import categorical_crossentropy


# Defining the 1st architecture
def neural_model1(num_input_nodes=281, num_label_nodes=4):
    model = Sequential()
    # Adding the first hidden layer with 5*node+5 neurons
    model.add(Dense(num_input_nodes, activation='relu', input_shape=(num_input_nodes,)))
    model.add(Dropout(0.2))
    # Add the remaining hidden layers
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    # Adding the output layer
    # The activation for the output layer is always set to sigmoid function in this exercise session
    model.add(Dense(num_label_nodes, activation='softmax'))

    model.summary()

    return model


# Defining the 2nd architecture
def neural_model2(num_input_nodes=281, num_label_nodes=4):

    model = Sequential()
    # Adding the first hidden layer with 5*node+5 neurons
    model.add(Dense(num_input_nodes, activation='relu', input_shape=(num_input_nodes,)))
    model.add(Dropout(0.2))
    # Add the remaining hidden layers
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    # Adding the output layer
    # The activation for the output layer is always set to sigmoid function in this exercise session
    model.add(Dense(num_label_nodes, activation='softmax'))

    model.summary()

    return model


# Defining the 3rd architecture
def neural_model3(num_input_nodes=281, num_label_nodes=4):

    model = Sequential()
    # Adding the first hidden layer with 5*node+5 neurons
    model.add(Dense(num_input_nodes, activation='relu', input_shape=(num_input_nodes,)))
    model.add(Dropout(0.2))
    # Add the remaining hidden layers
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    # Adding the output layer
    # The activation for the output layer is always set to sigmoid function in this exercise session
    model.add(Dense(num_label_nodes, activation='softmax'))

    model.summary()

    return model


# Defining the 4th architecture
def neural_model4(num_input_nodes=281, num_label_nodes=4):

    model = Sequential()
    # Adding the first hidden layer with 5*node+5 neurons
    model.add(Dense(num_input_nodes, activation='relu', input_shape=(num_input_nodes,)))
    model.add(Dropout(0.2))
    # Adding the output layer
    # The activation for the output layer is always set to sigmoid function in this exercise session
    model.add(Dense(num_label_nodes, activation='softmax'))

    model.summary()

    return model

