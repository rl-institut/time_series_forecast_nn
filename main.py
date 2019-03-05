"""
This project is a template to predict time series via a neural network.

Therefore training time series of the input data and the desired output have to be loaded. The neural network is created
with keras (https://keras.io/) and a tensorflow backend (https://www.tensorflow.org/).

The parameters to set up the template are given in the first section. The second section contains the code. Examples
in the comments try to explain each step so that the template can be adjusted if needed.

In this template, an example of the electricity price prediction based on the power of renewables in the grid and the
electricity consumption data is used. The data is divided into input and output data:

Input data: Data used for the prediction, later on called x_data (e.g. power renewables and demand data).
Output data: The data that should be predicted, later on called y_data (e.g. intraday electricity price).

To define how the input and the output of the neural net are connected, the time series data can be connected with
different offsets. An overview of the offsets in given in the following, where the data sets compared are
DATA SET IN 1, DATA SET IN 2 and DATA SET OUT:

                    | --- step distance --------| ~~~ DATA SET IN 1 2 ~ |
                    | ~~~ DATA SET IN 1 1 ~ |
        | - start - | --- horizon input 1 - |
INPUT1  |  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24| 25| 26| ...

                    | --- step distance --------| ~~~ DATA SET IN 2 2 ~~~~~~~~~ |
                    | ~~~ DATA SET IN 2 1 ~~~~~~~~~ |
        | - start - | --- horizon input 2 --------- |
INPUT2  |  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24| 25| 26| ...

                                                                        | ~~~ DATA SET OUT 2 ~~~~~~~~~~ |
                                            | --- step distance --------| --- horizon output ---------- |
                                            | ~~~ DATA SET OUT 1 ~~~~~~~~~~ |
        | - start - | --- horizon_offset -- | --- horizon output ---------- |
OUTPUT  |  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24| 25| 26| ...

"""

import os
import csv
import re
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                          PARAMETERS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Define the names of the input and output .csv time series files that have to be copied to the "time_series" folder.
# Multiple input data of interest can be used, each having an own column in the input time series .csv file.
input_ts = 'ts_renewable_load_2015.csv'
output_ts = 'day_ahead_eex_price_2015.csv'

# Define how long one batch size should be [time steps]
horizon_input = [24, 24]
horizon_output = 24

# Define an offset between the input and output data (this will lead to a prediction of the future) [time step].
horizon_offset = 0

# Define the distance between two training steps [time steps]
step_distance = 4

# Define the starting point [time_step]
start = 0

# Define the share of the data that should be used as training data (0.8 -> 80 %) [-]
share_training = 0.9

# Normalize data (if none is chosen, the min and max value of the time series are used for min and max val).
# The values for the input have to be in a list (while there can be more than one input).
input_min = [None, None]
input_max = [None, None]

output_min = None
output_max = None

# Define if the test data should be animated.
is_animation_wanted = True

# Define the number of epochs (how many time the training data should be used for training).
n_epoch = 15

# Define the activation function type of the two hidden layers (often used are 'relu' -> tf.nn.relu,
# 'sigmoid' -> tf.nn.sigmoid, 'tanh' -> tf.nn.tanh)
activation_fun = tf.nn.relu
# Define how many nodes the hidden layers (and if used GRU) should have.
n_node = 512

# Decide if a Gated Recurrent Unit (GRU) network is used (that is a network that can "remember" old states and is good
# if dependencies over longer periods of time are expected.
is_gru_used = False

# GRU PARAMETERS (JUST USED IF is_gru_used IS SET TO True
# Define the number of steps that should be aggregated in one batch.
batch_len = 30
# Define the number of steps between two sequential batches.
batch_step = 5
# Define warm up steps, which is a period at the beginning of a batch whose results are not considered in the loss
# function. This can be used to improve the accuracy of GRU networks, while the first guesses are usually bad.
# E.g. if the batch length (batch_len) is 30 and the warm up steps (warmup_step) is 5, only the entries 6-30 are
# considered by the loss function.
warmup_step = 0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                              CODE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


""" LOAD TIME SERIES """
# Load the time series of input and output.
# Generate the path to the .csv files in the "time_series" folder.
ts_folder = 'time_series'
path_input = os.path.join(ts_folder, input_ts)
path_output = os.path.join(ts_folder, output_ts)

# Load the input file.
input_data = []
with open(path_input, 'r') as file:
    file_data = csv.reader(file, delimiter=',')
    for row in file_data:
        # Check if this row contains text (character a-z) and therefore has to be a header. If so, skip this row.
        if not re.search("[a-z, A-Z]", row[0]):
            this_row = []
            for row_entry in row:
                this_row.append(float(row_entry))

            input_data.append(this_row)

input_data = np.array(input_data)
# Get the number of input data.
n_input = input_data.shape[1]

# Load the output file.
output_data = []
with open(path_output, 'r') as file:
    file_data = csv.reader(file, delimiter=',')
    for row in file_data:
        # Check if this row contains text (character a-z) and therefore has to be a header. If so, skip this row.
        if not re.search("[a-z, A-Z]", row[0]):
            # Output data can only have one column in the .csv file as input, this is checked here.
            if len(row) > 1:
                raise ValueError('Output .csv file may only have one column')

            output_data.append(float(row[0]))

output_data = np.array(output_data)

print('Time series have been loaded.')

""" NORMALIZE TIME SERIES """
# INPUT DATA
# In case of only one input data the min and max values might be set as not a list as well and thus have to be
# transformed to a list in this case.
input_min = input_min if type(input_min) == list else [input_min]
input_max = input_max if type(input_max) == list else [input_max]
for i_input in range(n_input):
    if input_min[i_input] is None:
        # Case: For this input data the lowest time series value should be used for normalization.
        input_min[i_input] = min(input_data[:, i_input])
    if input_max[i_input] is None:
        # Case: For this input data the highest time series value should be used for normalization.
        input_max[i_input] = max(input_data[:, i_input])
    # Normalize this input time series.
    input_data[:, i_input] -= input_min[i_input]
    input_data[:, i_input] /= (input_max[i_input] - input_min[i_input])

# OUTPUT DATA
if output_min is None:
    output_min = min(output_data)
if output_max is None:
    output_max = max(output_data)
# Normalize the output data.
output_data -= output_min
output_data /= (output_max - output_min)

""" SHAPE DATA FOR NEURAL NETWORK USAGE """
# Calculate the max. horizon.
max_horizon_input = max(horizon_input) if len(horizon_input) > 1 else horizon_input
max_horizon = max(max_horizon_input, horizon_output + horizon_offset)
# Calculate the max. number of sample data for the network.
length_timeseries = len(output_data)
n_sample_max = math.floor((length_timeseries - start - max_horizon)/step_distance) + 1

x_data = np.zeros((n_sample_max, sum(horizon_input)))
y_data = np.zeros((n_sample_max, horizon_output))
for i_step in range(n_sample_max):
    # Calculate the start index in the time series for this step.
    i_start = start + i_step * step_distance
    # X-DATA
    for i_input in range(n_input):
        this_input_data = input_data[i_start:i_start+horizon_input[i_input], i_input]

        if i_input == 0:
            this_x_data = this_input_data
        else:
            this_x_data = np.append(this_x_data, this_input_data)

    x_data[i_step, :] = this_x_data
    # Y-DATA
    this_output_data = output_data[i_start+horizon_offset:i_start+horizon_offset+horizon_output]
    y_data[i_step, :] = this_output_data

# Split the x and y data into training and test data.
n_train_data = math.floor(n_sample_max * share_training)
x_train = x_data[:n_train_data, :]
x_test = x_data[n_train_data:, :]
y_train = y_data[:n_train_data, :]
y_test = y_data[n_train_data:, :]

print('Feed in data for the neural net was created.')


# A generator for batches of training data.
def batch_gen(data, _batch_len=1, _batch_step=None, n_batch=None):
    # This batch generator splits the processed time series data into training batches for a GRU neural network.
    # Input:
    #  data: The time series data with size (1. number of steps, 2. data per step)
    #  batch_len: How many steps are going into one batch
    #  batch_step: How many steps lie between two subsequent batches
    #  n_batch: Number of batches given out

    n_data = data.shape[0]
    # Calculate how far a batch step can be to achieve the n_batch.
    if _batch_step is None and n_batch is None:
        raise ValueError('Either batch_step or n_batch has to be defined')
    elif n_batch is None:
        # If number of batches is not defined, create as many as possible.
        n_batch = 1 + math.floor((n_data-_batch_len)/_batch_step)
    elif n_batch == 1:
        # If n_batch equals one, the batch step does not matter but would lead to a division by zero in the next elif
        # statement, thus this is checked here.
        _batch_step = 0
    elif _batch_step is None:
        # If no batch step is chosen, split it as uniformly (wide) as possible.
        _batch_step = math.floor((n_data - _batch_len) / (n_batch-1))

    # Splitting up the data into batches
    batch_data = np.zeros((n_batch, _batch_len, data.shape[1]))
    for i_batch in range(n_batch):
        batch_data[i_batch, :, :] = data[i_batch*_batch_step:i_batch*_batch_step+_batch_len, :]

    return batch_data


""" LOSS FUNCTION AND METRICS """
# LOSS FUNCTION (DEFINES HOW WELL THE NETWORK IS DOING DURING THE TRAININGS PROCESS)
def custom_loss_fun(y_true, y_pred):
    # The loss function uses the mean squared error of predicted and real output.
    # Shape of the input tensors y_true and y_pred are:
    # [batch_size, sequence_length, num_y_signals].

    if is_gru_used:
        y_true = y_true[:, warmup_step:, :]
        y_pred = y_pred[:, warmup_step:, :]

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

# METRICS (VALUES THAT CAN BE DISPLAYED DURING THE TRAINING PROCESS)
def pred_max(y_true, y_pred):
    # Display the max. predicted output value.
    return K.max(y_pred)

def pred_min(y_true, y_pred):
    # Display the min. predicted output value.
    return K.min(y_pred)

def actual_max(y_true, y_pred):
    # Display the actual max. output value.
    return K.max(y_true)

def actual_min(y_true, y_pred):
    # Display the actual min. output value.
    return K.min(y_true)


""" DEFINE NEURAL NETWORK """
# There are two types of models, the sequential model is a linear stack of network layers.
model = tf.keras.models.Sequential()
# First layer is the input layer, which should be flat.
# model.add(tf.keras.layers.Flatten())
# When using a LSMT or GRU as input layer, a 3 dimensional input is required. The dimensions stand for 1. sample,
# 2. time step and 3. feature. If we want to just use one training set, this would mean 1 sample, thus the first
# dimension has to be a singelton. The second dimensions has an entry for each timestep, the third has an entry for each
# feature, meaning the actual input data.

if is_gru_used:
    # Case: GRU is used. Therefore the training data has to be divided into batches.
    x_train = batch_gen(x_train, batch_len, batch_step, None)
    y_train = batch_gen(y_train, batch_len, batch_step, None)
    x_test = batch_gen(x_test, batch_len, batch_step, None)
    y_test = batch_gen(y_test, batch_len, batch_step, None)
    # First layer of the network is a GRU layer.
    model.add(tf.keras.layers.GRU(units=n_node,
                                  return_sequences=True,
                                  input_shape=x_train.shape[1:]))

else:
    # Case: No GRU network should be used, use a flat layer as first layer of the network.
    model.add(tf.keras.layers.Flatten())

# Hidden layers (how many should be chosen depends on the problems complexity)
model.add(tf.keras.layers.Dense(n_node, activation=activation_fun))
model.add(tf.keras.layers.Dense(n_node, activation=activation_fun))
# Output layer (needs the size of the possible outputs, the activation should be probablistic, thus softmax is used)
# The number of output values depends on how many time steps in the future we want to look.
if is_gru_used:
    n_output_val = y_train.shape[2]
else:
    n_output_val = y_train.shape[1]
model.add(tf.keras.layers.Dense(n_output_val, activation=tf.keras.activations.linear))

''' TRAIN NEURAL NETWORK '''
# Parameters for the training of the model.
model.compile(optimizer='adam',
              loss=custom_loss_fun,
              metrics=['accuracy', pred_min, pred_max, actual_min, actual_max])

print('Start fitting of the neural net.')

# FIT THE MODEL
model.fit(x_train, y_train, epochs=n_epoch)

# Save the trained model.
model.save('test_model.model')

print('Fitting was successful.')


""" CALCULATE LOSS VALUE IN TEST DATA SET """
result = model.evaluate(x=x_test, y=y_test)
print('The loss of the test data set is {:.3}'.format(result[0]))

""" ANIMATE THE RESULT """
if is_animation_wanted:
    # Set up the plot.
    fig, ax = plt.subplots()
    ax.set(xlim=(0, horizon_output), ylim=(output_min, output_max), xlabel='steps', ylabel='Output value')
    steps = range(horizon_output)

    ln1 = ax.step([], [], label='Predicted output')[0]
    ln2 = ax.step([], [], label='Actual output', linestyle='--')[0]
    lines = [ln1, ln2]
    ax.legend(loc='lower center', shadow=True, fontsize='x-large')

    def update_animation(frame):
        # This function defines how a frame of the animation should look like.
        if is_gru_used:
            # For a GRU network, the data was divided into batches. The last batch of the test data will be animated.
            # For the output, the normalized data has to be computed back to the actual values.

            # Define the batch of the test data that should be animated.
            batch_to_animate = -1
            # Get the actual output of the batch to animate.
            this_actual_result = y_test[batch_to_animate, frame, :] * (output_max - output_min) + output_min
            # Calculate the prediction of one batch.
            this_predicted_result = model.predict(np.expand_dims(x_test[batch_to_animate, :, :], axis=0))
            this_predicted_result = this_predicted_result[0, frame, :] * (output_max - output_min) + output_min

        else:
            # If no GRU network was used, all the test data will be animated.
            # To use the model for predictions, the data has be fed in the shape (1, number of inputs per step).
            _this_input_data = x_test[frame].reshape((1, *x_test[frame].shape))
            # The data has to be brought back from the normalized form to its actual values.
            this_predicted_result = model.predict(_this_input_data) * (output_max - output_min) + output_min
            this_actual_result = y_test[frame, :] * (output_max - output_min) + output_min

        # Update the plot data.
        lines[0].set_data(steps, this_predicted_result)
        lines[1].set_data(steps, this_actual_result)
        ax.set_title('Frame ' + str(frame + 1))
        return lines

    if is_gru_used:
        n_frame = batch_len
    else:
        n_frame = x_test.shape[0]
    # Start the animation.
    ani = FuncAnimation(fig, update_animation, frames=n_frame, interval=1000, blit=False)
    plt.show()




































