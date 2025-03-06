#!/usr/bin/env python
# coding: utf-8

# ## Visualisation of the Feature Engineering Process (via Convolutional Filters)

# ### Key: Cathode = 0, Gate = 1, Tritium = 2

# # Imports

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random


# In[3]:


import tensorflow as tf
from tensorflow import keras
import seaborn as sn


# In[4]:


import pandas as pd
import awkward as ak
import pyarrow.parquet as pq

import hist
from hist import Hist, axis

import matplotlib as mpl
import matplotlib.patches as patches


# In[5]:


plt.rcParams['figure.figsize'] = [10, 8]
font = {'weight' : 'normal','size'   : 22}
plt.rc('font', **font)
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast

# Load data from Parquet file and convert to strings
data_path = '/Users/laith_mohajer/Documents/GitHub/MSCIProject/LZ Datasets/padded_waveforms.parquet'
df = pd.read_parquet(data_path)
print(df.head())
arr = ak.from_parquet(data_path) #Awkward DataFrame also created for future use
print(arr.fields) #analogous to df.columns.tolist()


# # Normalising the Data
# 

# In this instance, we use awkward arrays. Awkward arrays work very similarly to numpy arrays but they can have different lengths – for example, the times and samples arrays are a different length for every event.

# In[6]:


def normalise_array(arr):
    # Normalise each sub-array individually and return as an awkward array
    return ak.Array([
        (sub_array - np.min(sub_array)) / (np.max(sub_array) - np.min(sub_array)) if np.max(sub_array) != np.min(sub_array) else sub_array
        for sub_array in ak.to_list(arr)
    ])

def remove_array_offset(arr):
    # Normalise each sub-array individually and return as an awkward array
    return ak.Array([
        (sub_array - np.min(sub_array)) if np.max(sub_array) != np.min(sub_array) else sub_array
        for sub_array in ak.to_list(arr)
    ])

# Apply normalisation to each column
normalised_times = normalise_array(arr['times'])
normalised_samples = normalise_array(arr['samples'])
normalised_areas = normalise_array(arr['area'])

unnormalised_times = remove_array_offset(arr['times']) # Unnormalised times but shifted to start at zero (offset remove)

print(f"Lengths of first few subarrays in normalised_times: {ak.num(normalised_times, axis=1)[:10]}")
print(f"Example subarray from normalised_times: {normalised_times[0]}")
print(f"Example subarray from unnormalised_times: {unnormalised_times[3654]}")


# Print minimum and length for testing
# print(f"Minimum values of sub-arrays: {[sub_array.min() for sub_array in padded_times]}")
# print(f"Length of sub-array 200: {len(padded_times[200])}")


# # Standardising Length of Data

# In[7]:


# Lets pad the time data first using Awkward jargon

# first, an initial check to see if data row entry (training example) has an associated label. filtering for both NaNs and None values.
missing_label_count = len(arr[(arr['label'] != 0) & (arr['label'] != 1) & (arr['label'] != 2)])
print("Number of rows with unexpected or missing labels:", missing_label_count)

nan_areas = len(arr[ak.is_none(arr['area'])])
print("Number of NaN values in area column:", nan_areas)
# print(type(electron_size))

# Convert columns to numpy arrays, pad, and convert back to Awkward Arrays
def pad_to_max_length(array, max_length):
    return ak.Array(
        np.array(
            [np.pad(sub_array, (0, max_length - len(sub_array)), mode='constant', constant_values=0) for sub_array in ak.to_list(array)]
        )
    )

print(len(arr['times']))
print(f"Length Before Padding: {len(arr['times'][0])}")
print("Structure of 'times':", ak.type(arr['times']))

times_lengths = ak.num(arr['times'], axis=1)
max_time_length = ak.max(times_lengths) # returns length of longest time series in dataset
max_time_length_index = ak.argmax(times_lengths)

print(f'Results are: \n Max. Length = {max_time_length} \n Max. Length Index = {max_time_length_index}')

# Apply padding
padded_times = np.array(pad_to_max_length(normalised_times, max_time_length))
unnormalised_padded_times = np.array(pad_to_max_length(unnormalised_times, max_time_length))

sampling_interval_us = 0.01 # interval between consecutive samples in microseconds
time_us = [np.arange(len(wave)) * sampling_interval_us for wave in arr["times"]]
print(time_us[1036]) #TEST: the last element in this row should equal 18.3 µs

print(f"Length After Padding: {len(padded_times[0])}")
print(f"Length After Padding: {len(unnormalised_padded_times[3743])}")


# In this instance, we use awkward arrays. Awkward arrays work very similarly to numpy arrays but they can have different lengths – for example, the times and samples arrays are a different length for every event.

# Now lets standardise the Sample data

# In[8]:


print(len(arr['samples']))
print(f"Length Before Padding: {len(arr['samples'][0])}")
print("Structure of 'times':", ak.type(arr['samples']))

samples_lengths = ak.num(arr['times'], axis=1)
max_samples_length = ak.max(samples_lengths)
max_samples_length_index = ak.argmax(samples_lengths)

print(f'Results are: \n Max. Length = {max_samples_length} \n Max. Length Index = {max_samples_length_index}')

# Apply padding
padded_samples = np.array(pad_to_max_length(normalised_samples, max_samples_length))

print(f"Length After Padding: {len(padded_samples[0])}")
# print(padded_samples[0][-200:])


# In[9]:


fig, axs = plt.subplots(3, 1, figsize=(12, 12))  # axs is an array of Axes

# Plot the first dataset in the first subplot
axs[0].plot(arr['times'][32], arr['samples'][32], label='Raw Data', color='teal')
axs[0].set_title("Raw Data")
axs[0].set_ylabel("Amplitude")
axs[0].legend()

# Plot the second dataset in the first subplot
axs[1].plot(normalised_times[32], normalised_samples[32], label='Normalised Data', color='slateblue')
axs[1].set_title("Normalised Data")
axs[1].set_ylabel("Amplitude")
axs[1].legend()

# Plot the third dataset in the second subplot
axs[2].plot(padded_times[32], padded_samples[32], label='Normalised + Padded Data', color='deeppink')
axs[2].set_title("Normalised + Padded Data")
axs[2].set_ylabel("Normalised Amplitude")
axs[2].legend()

# Add overall labels
plt.xlabel("Time")

# Adjust spacing between subplots for readability
plt.tight_layout()

# Show the plot
plt.savefig('/Users/laith_mohajer/Documents/GitHub/MSCIProject/Figures/errorsinpaddingandnormalising.png')
plt.show()


# # Creating the Training and Test Data (AwkwardArrays)

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
# Define padding length
padding_length = 500

# X = arr[['times', 'samples']] #creates a mini array from mother array with only 'times' and 'samples' columns
#print(X)
y = np.array(arr['label']) # labelled as 0,1 and 2 corresponding to cathode, gate and tritium respectively. this is the true output data#

# Add zero-padding on each side of the data (only along the time dimension for 2D data) Then reshape X_train_padded and X_test_padded to 3D
normalised_times_padded = np.pad(padded_times, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
unnormalised_times_padded = np.pad(unnormalised_padded_times, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
normalised_samples_padded = np.pad(padded_samples, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
X = normalised_samples_padded
print(normalised_times_padded.shape)
print(X.shape)

time_steps = normalised_times_padded.shape[1]
example_index = 0
example_row = X[example_index]

# Split the row into time and samples
example_time = normalised_times_padded[example_index][500:-500]  # The time data of sample index {example_index}
example_samples = example_row[500:-500]  # The sample data of sample index {example_index}

plt.figure(figsize=(10, 6))
plt.plot(arr['times'][0], arr['samples'][0])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title(f"Unnormalised Input to the Model (Example {example_index})")

plt.show()
plt.plot(example_time, example_samples, label="Example Waveform")
plt.xlabel("Time")
plt.ylabel("Normalised Amplitude")
plt.title(f"Actual Input to the Model (Example {example_index})")
plt.legend()
# plt.grid(True)
plt.show()


# # S2 Area Spectrum

# In[11]:


areas = arr['area']
max_area = max(areas)

bins=100
electron_size = 58.5
print(max(areas)/electron_size)

# create a histogram of area distributions for gate, tritium and cathode data.
# first, boolean masks to filter gate, tritium and cathode data from main Awkward Array.
gate_events = arr[(arr['label'] == 1)].area / electron_size
tritium_events = arr[(arr['label'] == 2)].area / electron_size
cathode_events = arr[(arr['label'] == 0)].area / electron_size

gate_hist = Hist(hist.axis.Regular(bins,0,max(areas)/electron_size*1.01))
gate_hist.fill(gate_events)

tritium_hist = Hist(hist.axis.Regular(bins,0,max(areas)/electron_size*1.01))
tritium_hist.fill(tritium_events)

cathode_hist = Hist(hist.axis.Regular(bins,0,max(areas)/electron_size *1.01))
cathode_hist.fill(cathode_events)

print(tritium_hist.view())
print(gate_hist.view())
print(cathode_hist.view())


# In[12]:


bin_edges = gate_hist.axes[0].edges  # Get bin edges from one of the histograms
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
print(bin_edges[-2])

# Retrieve and adjust counts for each histogram
gate_counts = gate_hist.view() / bin_centers
tritium_counts = tritium_hist.view() / bin_centers
cathode_counts = cathode_hist.view() / bin_centers

print(cathode_hist.view().sum())
print(tritium_counts[-1])
print(gate_counts[-1])
print(cathode_counts[-1])


# # Weighting the S2 Area Spectrum

# In[13]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7), sharex=True, sharey=True)  #Initialise new fig object to plot weighted, flat spectrum
fig.subplots_adjust(hspace=0.0)

areas = arr['area']

max_tritium_area = max(areas[arr.label == 2])
max_gate_area = max(areas[arr.label == 1])
max_cathode_area = max(areas[arr.label == 0])
max_area = max(areas)
print(f'MAX areas: tritum,gate, cathode, overall dataset {max_tritium_area, max_gate_area, max_cathode_area, max_area}')

# Calculate the target flat spectrum as the average of the three histograms
gate_data = gate_hist.view(flow=False)
tritium_data = tritium_hist.view(flow=False) 
cathode_data = cathode_hist.view(flow=False) 

print(gate_counts.sum()) # total area under the histogram (integral of histogram)
print(tritium_counts.sum()) # total area under the histogram (integral of histogram)
print(cathode_counts.sum()) # total area under the histogram (integral of histogram)

average_counts = np.mean([gate_counts.sum(), tritium_counts.sum(),cathode_counts.sum()]) # average integral (area) of each histgoram to be used as reweighting benchmark

gate_weights = []
gate_reweighted  = []
for datapoint in range(len(gate_counts)):
    gate_reweighted.append(gate_counts[datapoint] * (1 / (gate_counts[datapoint] / gate_counts.sum())) if gate_counts[datapoint] != 0 else 0)
    gate_weights.append(1 / (gate_counts[datapoint] / gate_counts.sum()))

tritium_weights = []
tritium_reweighted = []
for datapoint in range(len(tritium_counts)):
    tritium_reweighted.append(tritium_counts[datapoint] * (1 / (tritium_counts[datapoint] / tritium_counts.sum())) if tritium_counts[datapoint] != 0 else 0)
    tritium_weights.append(1 / (tritium_counts[datapoint] / tritium_counts.sum()))

cathode_weights = []
cathode_reweighted = []
for datapoint in range(len(cathode_counts)):
    cathode_reweighted.append(cathode_counts[datapoint] * (1 / (cathode_counts[datapoint] / cathode_counts.sum())) if cathode_counts[datapoint] > cathode_counts[-1] else 0)
    cathode_weights.append(1 / (cathode_counts[datapoint] / cathode_counts.sum()) if datapoint < (len(cathode_counts)-1) else 0)

gate_reweighted = np.array(gate_reweighted)
tritium_reweighted = np.array(tritium_reweighted)

#tritium_reweighted[-1] = 37.92238004
cathode_reweighted = np.array(cathode_reweighted)

print(gate_reweighted)

#errorbars
gate_errors = np.sqrt(gate_counts) * gate_weights /60 # * (average_counts / gate_counts.sum())
tritium_errors = np.sqrt(tritium_counts) * tritium_weights /60 # * (average_counts / tritium_counts.sum())
cathode_errors = np.sqrt(cathode_counts) * cathode_weights/60 # * (average_counts / cathode_counts.sum())

# PADDING: Append zeros to make the histogram look like a "tophat"
tophat_padding = 5  # Number of zeros to add
gate_reweighted = np.append(gate_reweighted, [0] * tophat_padding)
tritium_reweighted = np.append(tritium_reweighted, [0] * tophat_padding)
cathode_reweighted = np.append(cathode_reweighted, [0] * tophat_padding)

gate_errors = np.append(gate_errors, [0] * tophat_padding)
tritium_errors = np.append(tritium_errors, [0] * tophat_padding)
cathode_errors = np.append(cathode_errors, [0] * tophat_padding)

# Extend bin_centers by adding extra bins to match the "tophat" shape
bin_step = bin_centers[1] - bin_centers[0]  # Calculate the step size between bins
extended_bin_centers = np.append(bin_centers, bin_centers[-1] + bin_step * np.arange(1, tophat_padding + 1))


# Apply reweighting by multiplying the original data by the weights

# Plot reweighted data with translucent error bands
# Gate
ax.plot(extended_bin_centers, gate_reweighted * (average_counts / gate_counts.sum()), color='slateblue', label='Gate (Reweighted)')
ax.fill_between(
    extended_bin_centers, 
    (gate_reweighted - gate_errors) * (average_counts / gate_counts.sum()), 
    (gate_reweighted + gate_errors) * (average_counts / gate_counts.sum()), 
    color='slateblue', alpha=0.3
)

# Tritium
ax.plot(extended_bin_centers, tritium_reweighted * (average_counts / tritium_counts.sum()), color='teal', label='Tritium (Reweighted)')
ax.fill_between(
    extended_bin_centers, 
    (tritium_reweighted - tritium_errors) * (average_counts / tritium_counts.sum()), 
    (tritium_reweighted + tritium_errors) * (average_counts / tritium_counts.sum()), 
    color='teal', alpha=0.3
)

# Cathode
ax.plot(extended_bin_centers, cathode_reweighted * (average_counts / cathode_counts.sum()), color='darkmagenta', label='Cathode (Reweighted)')
ax.fill_between(
    extended_bin_centers, 
    (cathode_reweighted - cathode_errors) * (average_counts / cathode_counts.sum()), 
    (cathode_reweighted + cathode_errors) * (average_counts / cathode_counts.sum()), 
    color='darkmagenta', alpha=0.3
)

ax.set(xlabel='Detected Electrons',ylabel='Events/electron')
#ax.set_yscale('log')
ax.legend(bbox_to_anchor=(1.5,0.9),loc='upper right',frameon=False,fontsize=14) 
plt.savefig('/Users/laith_mohajer/Documents/GitHub/MSCIProject/Figures/weighted_pulse_spectrum.png', dpi=1800)
plt.show()


# # Creating the Weight Array to Feed into CNN

# In[14]:


gate_weights = np.array(gate_weights)
gate_weights = np.where(np.isinf(gate_weights), 0, gate_weights) # check if values are defined (finite), 0 if not.

tritium_weights = np.array(tritium_weights)
tritium_weights = np.where(np.isinf(tritium_weights), 0, tritium_weights)

cathode_weights = np.array(cathode_weights)
cathode_weights = np.where(np.isinf(cathode_weights), 0, cathode_weights)

#gate_weights = gate_weights.ravel() # gate_weights was found to be a 2D array with only 1 row which prohibited proper indexing. ravel() flattens to 1D ndarray.
print(f'these are da {gate_weights.size}')

print(gate_data.sum())
print(tritium_data.sum())
print(cathode_data.sum())

def subdataset_total_weights(dataset_weights, n_data_per_bin):
    n_data_per_bin = np.array(n_data_per_bin, dtype=int) #creates a copy of array that is an ndarray with every element being an integer
    weight_list = []
    for i in range(bins):
        weight_list.extend([dataset_weights[i]] * n_data_per_bin[i])
    return np.array(weight_list)

g_weights = subdataset_total_weights(gate_weights, gate_data)
t_weights = subdataset_total_weights(tritium_weights, tritium_data)
c_weights = subdataset_total_weights(cathode_weights, cathode_data)

print(g_weights.size + t_weights.size + c_weights.size)
print(type(g_weights))
print(len(arr))


# # Creating and Populating the New Weight Column in the Main Dataframe

# In[15]:


weight_column_4_mainarray = np.zeros(len(arr))
# print(len(arr['weights'])) #check to ensure its same length as number of training examples

# Now we need to create an iterative loop that finds a cathode, gate or tritium training example and populates its assoicated weight with the correct weight parameter from its respective weight array

gate_event_counter = 0
cathode_event_counter = 0
tritium_event_counter = 0

print(len(t_weights))

for i in range(len(arr)):
    if arr['label'][i] == 0: # Cathode
        # print(g_weights[i])
        weight_column_4_mainarray[i] = c_weights[gate_event_counter]
        gate_event_counter += 1
    elif arr['label'][i] == 1: # Gate
        weight_column_4_mainarray[i] = g_weights[cathode_event_counter]
        cathode_event_counter += 1
    else: # Tritium (as we have already verified there are no None or NaN entries)
        weight_column_4_mainarray[i] = t_weights[tritium_event_counter]
        tritium_event_counter += 1

arr['weights'] = weight_column_4_mainarray # a new weight column has been initialised!


# # Optional Code: Normalising the Weights Array

# In[16]:


total_weight = ak.sum(arr['weights'])

# Normalize the weights by dividing each element by the total weight
normalized_weights = arr['weights'] / total_weight

# If you want to save the normalized weights back to the array. In Awkward, this is done by duplicating the original array and adding a new column.
arr = ak.with_field(arr, normalized_weights, 'weights_normalized')

print(arr['weights_normalized'])


# # Convolutional Neural Network

# In[17]:


seed_value = 42 # set a global random seed for model reproducibility
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping

weights_np = ak.to_numpy(arr['weights'])
normalised_weights_np = ak.to_numpy(arr['weights_normalized'])
print(len(weights_np))
# An issue arises here initially as arr['weights'] is an awkward array. Keras only recognises and deals with a NumPy array therefore conversion is neccessary
# Another issue also arises in that the test and train data do not have asscoated weights as the weights column was initialised after the split was made
# 'arr' is the original dataset
normalized_area = ak.to_numpy(arr['area'] / electron_size)  # converting 'area' to detected electrons by dividing by 58.5

labels = arr['label']

radii = ak.to_numpy(arr['r'])

# TO ALTER BETWEEN WEIGHTS AND NORMLAISED WEIGHTS> CHANGE WEIGHTS_NP VARIABLE ACCORDINGLY> DEFAULT: (UNNORMALISED) WEIGHTS
X_train, X_test, y_train, y_test, area_train, area_test, weights_train, weights_test, \
normalised_times_train, normalised_times_test, times_us_train, times_us_test, \
normalised_samples_train, normalised_samples_test, r_train, r_test = train_test_split(
    X, labels, normalized_area, weights_np, normalised_times, time_us, normalised_samples, radii, 
    test_size=0.25, random_state=42
)

print("Length of X_train:", len(X_train))
print("Length of y_train:", len(y_train))
print("Length of weights_train:", len(weights_train))
print("Shape of weights_train:", weights_train.shape)
print("Shape of X_train:", X_train.shape)

y_train = np.array(y_train) # this is neccessary as train_test_split often returns lists instead of ndarrays but Keras.model.fit requires the functionality of ndarrays
y_test = np.array(y_test)
print("Shape of y_train:", y_train.shape)

y_test_np = ak.to_numpy(y_test)
area_test_np = ak.to_numpy(area_test)

X_train_padded = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) # adding a channels dimension (greyscale of 1) to enable seamless input into CNN
X_test_padded = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[19]:


early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    min_delta=0.01,      # Minimum change in loss to qualify as an improvement
    patience=3,          # Stop training after 3 epochs of no significant improvement
    verbose=1,           # Print a message when stopping
    restore_best_weights=True  # Restore the weights from the best epoch
)

convoNN = keras.Sequential([
    # First 1D convolution layer
    keras.layers.Conv1D(filters=28, kernel_size=200, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    
    # Second 1D convolution layer
    keras.layers.Conv1D(filters=64, kernel_size=200, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    
    # Flatten layer to connect to dense layers. 2D pooled feauture map flattened to 1D vector to input into dense outer layers.
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # Adjust output size for the number of classes
])

# Compile the model
convoNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#CNN with 7 layers
#the first two Conv2D extract spatial features from the image- i.e. there will be 28 filters that will scan the image for patterns, each filter extracts different features from the image (e.g.textures, edges)
#max Pooling layer performs down-sampling to resduce spatial dimensions
#(2) specifies a pooling window which means the layer will take the max value from every 2-linear-unit-wide region
#flatten layer converts information from 2D map to a 1D vector
#64 layer is a dense layer with 64 neurons
#10 layer is our 10 neuron layer that provides a class (digit 0-1)

convoNN.fit(X_train_padded, y_train, sample_weight=weights_train, epochs=5, batch_size=218, validation_split=0.2, verbose=0)


# ### Model Evaluation on Test Data

# In[20]:


X_test_reshaped = X_test.reshape(-1, 5656, 1)
print(X_test_reshaped.shape)
convoNN.evaluate(X_test_padded, y_test, verbose=0) # tf.model.evaluate() will return a list (or tuple) with two numbers: 
# The first number is the loss computed on the test set (using sparse categorical crossentropy).
# The second number is the accuracy computed on the test set.


# # Visualising the Model Classification Effectiveness: The Probability Distribution

# In[27]:


# Step 1: Get predicted probabilities for each class
y_pred_proba = convoNN.predict(X_test_padded)

predicted_classes = np.argmax(y_pred_proba, axis=1) # axis = 1 signifies that argmax should parse through the columns of each row and find the max. value
# for i in range(30):
    # print(f"True label: {y_test[i]}, Predicted class: {predicted_classes[i]}")  # Initial checking to see how well the model predicts the classes for first 30 training examples 

# Step 2: Calculate background and bulk probabilities
background_proba = y_pred_proba[:, 0] + y_pred_proba[:, 1]  # Sum of Gate and Cathode probabilities
bulk_proba = y_pred_proba[:, 2]  # Tritium probabilities. All the training examples in the training dataset that the model has predicted to be tritium.

# Separate true Tritium events from non-Tritium events in y_test
true_tritium_mask = (y_test == 2) # actual tritium events
non_tritium_mask = (y_test != 2) # non-tritium(background) events

threshold = np.percentile(bulk_proba[true_tritium_mask], 20)

# Step 3: Plot histograms for both groups
plt.figure(figsize=(10, 6))

# Histogram for actual Bulk (Tritium) events
plt.hist(bulk_proba[true_tritium_mask], bins=30, alpha=0.5, color='red', label='True Tritium', density=True)

# Histogram for non-Bulk (Tritium) events
plt.hist(bulk_proba[non_tritium_mask], bins=30, alpha=0.3, color='blue', label='Non-Tritium', density=True)

# Add titles and labels
# plt.title('Predicted Probability Histogram for Tritium Classification')
plt.axvline(x=threshold, color='gray', linestyle='--', label=f'Analysis Threshold = {threshold:.2f}')
plt.xlabel('Predicted Probability of Tritium')
plt.ylabel('Density')
plt.legend(loc='upper center')

plt.show()


# # Visualising the Inputs to the Model

# In[22]:


from tensorflow.keras.models import Model
print(X_train_padded.shape[1])
print(y_train.shape)

# Select every 10^2-th waveform
step = 100  # Step size for selection
waveforms_to_plot = X_train_padded[::step]

num_samples = X_train_padded.shape[1]
print(num_samples)

# Calculate sample length before padding
original_time_sample_length = (num_samples - (2 * padding_length)) #// 2 # floor division (round down to nearest integer)
print(original_time_sample_length)
print(original_time_sample_length + (padding_length))

# Visualize the first 10 padded training examples
# Generate 10 random indices to plot 10 random examples
random_indices = np.random.randint(0, len(normalised_samples_train), size=10)

plt.figure(figsize=(12,12))
for idx, i in enumerate(random_indices):  # Iterate through the random indices
    plt.subplot(5, 2, idx + 1)
    plt.plot(times_us_train[i], normalised_samples_train[i])
    plt.title(f"Training Example Waveform {i}", fontsize=16)
    plt.xlabel("Time (µs)", fontsize = 16)
    # plt.ylabel("Amplitude (Normalised)")

plt.ylabel("Amplitude (Normalised)", fontsize=16)
plt.tight_layout()
plt.show()


# # Visualising the Feature Maps (Detected Features by the Filters)

# To extract numerical feature maps from these layers, we need to create an intermediate model ('feauture_model') and pass real input data through it.
# 'feauture_model' is a new sub-model that takes the same input as convoNN but outputs intermediate activations (feature maps) from the layers specified in layer_outputs.

# In[23]:


sample_index = 32


# In[24]:


plt.plot(times_us_train[32], normalised_samples_train[32]) # X_train padded is just the reshaped version to change the 2D array into 3D format - the format required for CNN input
# plt.scatter(X_train[0][:2828], X_train[0][2828:])
#plt.plot(arr['times'][0], arr['samples'][0])
# plt.plot(X[0][:1828], X_combined[0][1828:])
plt.ylabel("Amplitude (Normalised)")
plt.xlabel("Time (µs)")
plt.title('Training Example Waveform 32 as Inputted into CNN')
plt.show()

print(convoNN.summary())


# In[25]:


from tensorflow.keras.models import Model
import math

# Specify the layers from which you want to extract features (e.g., the outputs of the Conv1D layers)
layer_outputs = [layer.output for layer in convoNN.layers if isinstance(layer, keras.layers.Conv1D)]
feature_model = Model(inputs=convoNN.input, outputs=layer_outputs) # Creating the intermediate model. Same input as CNN but outputs the activations of certain layers

sample_input = X_test_padded[sample_index:sample_index+1]  # Selecting a specific example
# When you use slicing — even if the slice only contains one element — Python returns a subarray that retains the original dimensions. 
# So X_test_padded[32:33] gives you an array with shape (1, time_steps, channels). The batch dimension is preserved.
# Most deep learning frameworks, including Keras, expect the input data to have a specific shape that includes the batch dimension. 
# Typically, a model is defined to accept inputs of shape (batch_size, time_steps, channels) for a 1D convolution. 
# Even if predicting on a single sample, the model still expects a three-dimensional tensor:
# With the Batch Dimension: The input shape is (1, time_steps, channels), where 1 represents one sample in the batch. Keeps the tensor 3D - suitable for CNN input.

feature_maps = feature_model.predict(sample_input) # This variable is intended to hold the outputs (also called activations or feature maps) 
# produced by selected layers when an input is passed through the intermediate model.

for i, fmap in enumerate(feature_maps):
    num_filters = fmap.shape[-1]
    # In a convolutional layer’s output (the feature map), the data is typically stored in a multi-dimensional array (or tensor). 
    # For a 1D convolution, the common shape is ({batch_size}, {time_steps}, {num_filters}).
    # The expression fmap.shape[-1] accesses the size of the last dimension of the tensor (the number of filters (or channels) used in that convolutional layer).
    # By assigning this value to num_filters, you now know how many separate filters’ activations are present in the feature map (the depth of the map). 
    # Useful for when you want to loop through and visualise the output of each filter separately.

    # Determine grid dimensions. Here we choose 4 columns.
    ncols = 4
    nrows = math.ceil(num_filters / ncols)

    plt.figure(figsize=(15, nrows * 3))
    for j in range(num_filters):
        plt.subplot(nrows, ncols, j+1)
        # Plot the feature map for filter j from the first sample in the batch.
        plt.plot(fmap[0, :, j])
        plt.title(f'Filter {j+1}', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    
    # Set a super-title for the figure to indicate the layer number.
    plt.suptitle(f'Layer {i+1} Filter Maps', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# In[42]:


# Define the event types with their corresponding label numbers.
event_types = {
    "Cathode": 0,
    "Gate": 1,
    "Tritium": 2
}

# For each event type, select examples in a grid of (pulse area, radius) categories.
for event_name, event_label in event_types.items():
    # Get indices for this event type from y_train.
    event_indices = np.where(y_train == event_label)[0]
    
    # Get the pulse areas and radii for these indices.
    areas = area_train[event_indices]
    radii = r_train[event_indices]
    
    # Compute thresholds (33rd and 66th percentiles) for pulse area.
    area_small_thresh = np.percentile(areas, 33)
    area_large_thresh = np.percentile(areas, 66)
    # And for the radius.
    r_close_thresh = np.percentile(radii, 33)
    r_far_thresh = np.percentile(radii, 66)
    
    # Define lambda functions for categorizing pulse area.
    area_categories = {
        "Small": lambda x: x < area_small_thresh,
        "Medium": lambda x: (x >= area_small_thresh) & (x <= area_large_thresh),
        "Large": lambda x: x > area_large_thresh
    }
    # And for the radius.
    radius_categories = {
        "Close": lambda x: x < r_close_thresh,
        "Medium": lambda x: (x >= r_close_thresh) & (x <= r_far_thresh),
        "Far": lambda x: x > r_far_thresh
    }
    
    # Loop over each combination of pulse area category and radius category.
    for area_cat, area_func in area_categories.items():
        for radius_cat, radius_func in radius_categories.items():
            # Filter the indices for which both the pulse area and radius satisfy the category.
            cat_indices = [idx for idx in event_indices 
                           if area_func(area_train[idx]) and radius_func(r_train[idx])]
            if len(cat_indices) == 0:
                print(f"No {event_name} events for area {area_cat} and radius {radius_cat}.")
                continue  # Skip if there is no event matching this combination.
            
            # Randomly choose one sample index from the filtered list.
            sample_index = np.random.choice(cat_indices)
            
            print(f"\n{event_name} event - Area: {area_cat}, Radius: {radius_cat} (Sample index {sample_index})")
            
            # --- Process and Plot the Selected Sample ---
            # Extract the sample input (with a batch dimension) for the CNN.
            sample_input = X_train_padded[sample_index:sample_index+1]
            
            # Run the sample through the intermediate feature model.
            feature_maps = feature_model.predict(sample_input)
            
            # Print the feature map shapes for debugging.
            print("Feature map shapes:")
            for i, fmap in enumerate(feature_maps):
                print(f"  Conv Layer {i+1}: {fmap.shape}") 
            """
            The shape (B, T, F) of the feature map tells you about the structure of the output from a convolutional layer. Here's what each number represents:

            1. Batch Size:
            This indicates that the feature map comes from a single sample (one input) that was passed through the network. In many deep learning frameworks the output retains a batch dimension.

            2. Temporal Steps:
            This number represents the length of the output along the time axis (if you're working with time-series data) AFTER applying the convolution (and possibly pooling, padding, etc.), the output has T distinct positions (or time steps).

            3. Number of Filters (Channels):
            For each of the T positions, the layer produces F separate activation values — one from each filter. Each filter is responsible for detecting a different type of feature from the input data.
            """
            # Retrieve the original waveform and its time axis.
            waveform = normalised_samples_train[sample_index]
            time_axis = times_us_train[sample_index]  # Already converted to microseconds.
            
            # Compute the amplitude range of the waveform.
            waveform_min = np.min(waveform)
            waveform_max = np.max(waveform)
            waveform_range = waveform_max - waveform_min
            
            # Create a new figure.
            plt.figure(figsize=(12, 8))
            
            # Plot the original waveform.
            plt.plot(time_axis, waveform, label='Original Waveform', color='black', linewidth=2)
            
            # Define some colors for different convolutional layers.
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            # Loop over each convolutional layer's feature map.
            for i, fmap in enumerate(feature_maps):
                # Each fmap is of shape (1, time_steps_layer, num_filters).
                # Sum over the filter dimension (axis = -1) to aggregate activations.
                # Then [0] removes the batch dimension.
                combined_map = np.sum(fmap, axis=-1)[0]
                
                # Create a new time axis for the combined feature map.
                t_feature = np.linspace(time_axis[0], time_axis[-1], num=combined_map.shape[0])
                
                # Rescale the combined map:
                combined_shifted = combined_map - np.min(combined_map)
                if np.max(combined_shifted) != 0:
                    combined_normalized = combined_shifted / np.max(combined_shifted)
                else:
                    combined_normalized = combined_shifted
                scaled_combined = combined_normalized * waveform_range + waveform_min
                
                # Plot the scaled combined feature map.
                plt.plot(t_feature, scaled_combined, label=f'Conv Layer {i+1} Combined', 
                         linestyle='--', color=colors[i % len(colors)])
            
            # Add dummy entries to the legend for pulse area and radius.
            pulse_area = area_train[sample_index]
            event_radius = r_train[sample_index]
            plt.plot([], [], color='none', linestyle='none', marker='None',
                     label=f"Pulse area: {pulse_area:.2f} e⁻")
            plt.plot([], [], color='none', linestyle='none', marker='None',
                     label=f"Radius: {event_radius:.2f}")
            
            # Finalise the plot.
            plt.xlabel("Time (µs)")
            plt.ylabel("Amplitude (Normalised)")
            plt.title(f"{event_name} Event: {area_cat} Area, {radius_cat} Radius (Training Sample {sample_index})", fontsize=18)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.show()


# ## Aside: Dataset of Filter Activations

# In[ ]:


# We want to look at the activations from the first convolutional layer:
# shape of feature_maps[0] is (1, T, F)

# Remove the batch dimension:
activations = feature_maps[0][0]  # Now activations.shape is (T, F)

# Optionally, if you have a time axis (for example, t_feature) that matches the T time steps,
# you can use it as the index. Otherwise, we'll just use the row numbers.
T, F = activations.shape # [rows, columns]
# Create column names for the filters:
columns = [f"filter {i}" for i in range(F)]
# Create a DataFrame:
df = pd.DataFrame(activations, columns=columns)
# Optionally, set the index to represent the time steps if you have a corresponding time axis.
# For example, if t_feature is a numpy array of length T:
# df.index = t_feature

# Display the DataFrame:
print(df)


# ## Guided Backpropagation

# In[57]:


# 1. Define the custom guided ReLU with a modified gradient.
@tf.custom_gradient
def guided_relu(x):
    # Forward pass: standard ReLU.
    y = tf.nn.relu(x)
    def grad(dy):
        # Only allow positive gradients to flow back.
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
    return y, grad

# 2. Function to replace standard ReLU activations in a model with guided ReLU.
def modify_model_for_guided_backprop(model):
    for layer in model.layers:
        # Replace the activation function if it is ReLU.
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
    return model

# 3. Create a guided backpropagation model based on your existing CNN (e.g. convoNN).
guided_model = tf.keras.models.clone_model(convoNN)
guided_model.set_weights(convoNN.get_weights())
guided_model = modify_model_for_guided_backprop(guided_model)

# 4. Define a function to compute guided backprop gradients.
def compute_guided_backprop_gradients(input_data, model, target_class_index):
    # Convert input to a tensor and ensure it's being watched.
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        # Select the score for the target class.
        loss = predictions[:, target_class_index]
    # Compute gradients of the loss with respect to the input.
    gradients = tape.gradient(loss, input_tensor)
    return gradients

# 5. Example usage: Choose a sample and compute guided backprop.
sample_index = 3031  # For example, use sample index 32 from the training set.
sample_input = X_train_padded[sample_index:sample_index+1]  # Keep the batch dimension.
target_class = int(y_train[sample_index])  # Use the true label as the target class.

# Compute the guided backprop gradients.
guided_gradients = compute_guided_backprop_gradients(sample_input, guided_model, target_class)

# Since sample_input shape is (1, time_steps, channels), remove the batch dimension.
guided_gradients = guided_gradients[0, :, 0].numpy()  # Assuming a single channel.

# 6. Plot the original waveform with the guided backprop gradients.
waveform = X_train_padded[sample_index]
time_axis = times_us_train[sample_index] # not padded so lets just create a linspace as the sequence is monotonic.
new_time_axis = np.linspace(times_us_train[sample_index][0],
                            times_us_train[sample_index][-1],
                            num=len(guided_gradients))


print(len(waveform))

plt.figure(figsize=(12, 8))
plt.plot(new_time_axis, waveform, label='Original Waveform', color='black', linewidth=2)
plt.plot(new_time_axis, guided_gradients, label='Guided Backprop Gradients', color='red', linestyle='--')
plt.xlabel("Time (µs)")
plt.ylabel("Normalised Amplitude / Gradient")
plt.title(f"Guided Backpropagation for Sample {sample_index} (Target Class {target_class})")
plt.legend()
plt.tight_layout()
plt.show()

