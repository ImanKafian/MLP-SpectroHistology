'''
About: The main Python script to develop a classification program based on MLP neural networks,
       a spectroscopic dataset as the predictor variables, and an 1D histology score dataset as the target variable.
Author: Iman Kafian-Attari
Date: 20.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. Select the output directory.
1. Select the training X 2D numpy array.
1. Select the test X 2D numpy array.
1. Select the training Y 2D numpy array.
1. Select the test Y 2D numpy array.
=========================================================
Notes:
1. This code is meant to create a classification problem using the following:
   - different MLP neural networks,
   - a spectroscopic dataset as predictors,
   - a 1D histoloy score as the target.
2. It requires the following inputs from the user:
   - an output directory,
   - two numpy 2D matrices containing the information on the training and test datasets for the predictor in the form of mxn
     where m: number of observation and n: number of predictor variables,
   - two numpy 2D matrices containing the information on the training and test datasets for the target in the form of mx1
     where m: number of observation and 1: the only target variable,
3. It automatically creates the classification problem for four different MLP architectures.
4. It stores and plots the performance of each model on the training and test datasets.
=========================================================
TODO for version O.2
1. Modify the code in a functional form.
2. Modify to code to work for any number of target variables.
=========================================================
'''

print(__doc__)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
import pandas as pd

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from tensorflow.keras.losses import categorical_crossentropy

from architectures.neural_network_models import neural_model1, neural_model2, neural_model3, neural_model4

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

output_dir = filedialog.askdirectory(parent=root, initialdir='C:\\', title='Select the output directory')

# Reading the predictors and references
x_train = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the training input file, a 2D numpy array'))
x_test = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the test input file, a 2D numpy array'))
y_train = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the training output file, a 2D numpy array'))
y_test = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the test output file, a 2D numpy array'))
y_test1 = y_test

# Reading the range of categorical histology score:
labels = ','.split(input('Please insert the range of labels used for the target score, separated with a comma (,),'
                         ' e.g. 0,1,2,3,4 --> '))

# Dimension of the train set
dim_x_input, dim_y_input = x_train.shape

# Normalizing the targeted data to a categorical dataset
y_train = keras.utils.to_categorical(y_train, len(labels))
y_test = keras.utils.to_categorical(y_test, len(labels))

# Creating a Pandas dataframe to store the performance of the NN models
performance = {'loss': [],
               'accuracy': []}
architecture_report = {'NN1': {}, 'NN2': {}, 'NN3': {}, 'NN4': {}}

# Compiling and fitting the model based on the 1st neural network architecture
models = [neural_model1(dim_y_input), neural_model2(dim_y_input), neural_model3(dim_y_input), neural_model4(dim_y_input)]
for architecture in range(len(models)):
    model = models[architecture]
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=60, epochs=500, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)

    # Visualizing the performance of the model
    print(f'\n Architecture {architecture+1} Performance')
    print(f'Total loss: {score[0]}')
    performance['loss'].append(score[0])
    print(f'Total accuracy: {score[1]*100}')
    performance['accuracy'].append(score[1]*100)

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train accuracy', 'test accuracy'], loc='best')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train loss', 'test loss'], loc='best')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.suptitle(f'Performance of Architecture {architecture+1}')
    plt.savefig(f'{output_dir}\\PerformanceArchitecture{architecture+1}.png', dpi=300)
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    # Prediction
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    pred_acc = confusion_matrix(y_test1.ravel(), y_pred, labels=list(labels))
    print(pred_acc)
    report = classification_report(y_test1.ravel(), y_pred, labels=list(labels), output_dict=True, zero_division=0)
    architecture_report[f'NN{architecture+1}'] = report
    print(report)

performance = pd.DataFrame(performance, index=['NN1', 'NN2', 'NN3', 'NN4'])
performance.to_csv(f'{output_dir}\\ArchitecturePerformance.csv', sep='\t')
print(performance)

architecture_report = pd.DataFrame.from_dict(architecture_report)
architecture_report.to_csv(f'{output_dir}\\ArchitectureClassificationReport.csv', sep='\t')
print(architecture_report)
