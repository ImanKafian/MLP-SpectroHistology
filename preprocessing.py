'''
About: Python script to preprocess the predictor and target matrices for the MLP classification program.
Author: Iman Kafian-Attari
Date: 20.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. Select the output directory.
2. Select the file containing information on the predictor matrix.
3. Select the file containing information on the target matrix.
=========================================================
Notes:
1. This script is meant to create the training and test datasets for the MLP neural network for the classification problem.
2. This script must be executed before running the main script.
3. It requires the following inputs from the user:
   - an output directory,
   - a numpy 2D matrix containing the information on the predictor in the form of mxn
     where m: number of observation and n: number of predictor variables,
   - a numpy 2D matrix containing the information on the predictor in the form of mx1
     where m: number of observation and 1: the only target variable,
4. It randomly creates the training and test sets for the predictor and target variables,
5. It transfer the range of values for the predictor variables to the range of [0, 1],
6. It stores the following data:
   - x_train,
   - x_test,
   - y_train,
   - y_test,
7. The output files are saved as a numpy 2D arrays.
8. To use this program without any errors, the target variables should be in the form of mx1 where m: number of samples.
=========================================================
TODO for version O.2
1. Modify the code in a functional form.
2. Modify to code to work for any number of target variables.
=========================================================
'''

print(__doc__)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

output_dir = filedialog.askdirectory(parent=root, initialdir='C:\\', title='Select the output directory')

# Import the data
# PREDICTORS:
us = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the input file, a 2D numpy array'))

# REFERENCES:
cells = np.loadtxt(filedialog.askopenfilename(parent=root, initialdir='C:\\', title='Select the output file, a 2D numpy array'))

# Normalizing the data into [0, 1]
scaler = MinMaxScaler()
us = scaler.fit_transform(us)

# Making the train and test set
x_train, x_test, y_train, y_test = train_test_split(us, cells, test_size=0.25)
np.savetxt(f'{output_dir}\\x_train.txt', x_train, delimiter='\t')
np.savetxt(f'{output_dir}\\x_test.txt', x_test, delimiter='\t')
np.savetxt(f'{output_dir}\\y_train.txt', y_train, delimiter='\t')
np.savetxt(f'{output_dir}\\y_test.txt', y_test, delimiter='\t')
