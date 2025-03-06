# Hyperparam Optimisation Research Focussing on Learning Rate, Batch-Size, Stride, N_filters, Pool Size, Kernel Size, Front and Back Trimming
# Key: Cathode = 0, Gate = 1, Tritium = 2
# Key: 1 timestep = 10 ns

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import awkward as ak
import hist
from hist import Hist, axis
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import legacy
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Define standard plot colors and style for consistency
plot_colors = {
    'gate': 'slateblue',
    'tritium': 'teal',
    'cathode': 'darkmagenta',
    'general': 'black',
    'training': 'forestgreen',
    'test': 'firebrick'
}

# Set standard matplotlib style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.grid'] = False

# Define the base directory for saving figures
base_dir = '/Users/laith_mohajer/Documents/GitHub/MSCIProject/'

# Function to create consistent plots
def create_standardized_plot(x, y, xlabel, ylabel, title, label=None, color='black', marker='o', linestyle='-', figsize=(12, 8)):
    plt.figure(figsize=figsize)
    plt.plot(x, y, marker=marker, color=color, linestyle=linestyle, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    if label:
        plt.legend()
    plt.tight_layout()
    return plt

# Plot settings
plt.rcParams['figure.figsize'] = [10, 8]
font = {'weight': 'normal', 'size': 22}
plt.rc('font', **font)

# Load data from Parquet file and convert to strings
data_path = f'{base_dir}/LZ Datasets/padded_waveforms.parquet'
df = pd.read_parquet(data_path)
print(df.head())
arr = ak.from_parquet(data_path)
print(arr.fields)

# Normalising the Data
def normalise_array(arr):
    min_val = ak.min(arr, axis=-1)
    max_val = ak.max(arr, axis=-1)
    return (arr - min_val) / (max_val - min_val)

# Apply normalization to each column
normalised_times = normalise_array(arr['times'])
normalised_samples = normalise_array(arr['samples'])
normalised_areas = normalise_array(arr['area'])

# Print or inspect the results
print(normalised_times)
print(normalised_samples)

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


padded_times = np.array(pad_to_max_length(normalised_times, max_time_length))
unnormalised_times = arr['times']
unnormalised_padded_times = np.array(pad_to_max_length(unnormalised_times, max_time_length))

sampling_interval_us = 0.01 # interval between consecutive samples in microseconds (1 timestep : 10 ns)
time_us = [np.arange(len(wave)) * sampling_interval_us for wave in unnormalised_times]
print(time_us[1036]) #TEST: the last element in this row should equal 18.3 µs

print(f"Length After Padding: {len(padded_times[0])}")
print(f"Length After Padding: {len(unnormalised_padded_times[3743])}")

# Applying now to sample data
print(len(arr['samples']))
print(f"Length Before Padding: {len(arr['samples'][0])}")
print("Structure of 'times':", ak.type(arr['samples']))

samples_lengths = ak.num(arr['times'], axis=1)
max_samples_length = ak.max(samples_lengths)
max_samples_length_index = ak.argmax(samples_lengths)

print(f'Results are: \n Max. Length = {max_samples_length} \n Max. Length Index = {max_samples_length_index}')

# Apply initial padding to standardise the length of all samples
padded_samples = np.array(pad_to_max_length(normalised_samples, max_samples_length))

print(f"Length After Padding: {len(padded_samples[0])}")

padding_length = 500

# X = arr[['times', 'samples']] #creates a mini array from mother array with only 'times' and 'samples' columns
#print(X)
y = np.array(arr['label']) # labelled as 0,1 and 2 corresponding to cathode, gate and tritium respectively. this is the true output data#

# Add zero-padding on each side of the data (only along the time dimension for 2D data) Then reshape X_train_padded and X_test_padded to 3D
normalised_times_padded = np.pad(padded_times, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
unnormalised_times_padded = np.pad(unnormalised_padded_times, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
normalised_samples_padded = np.pad(padded_samples, ((0, 0), (padding_length, padding_length)), mode='constant', constant_values=0)
X = normalised_samples_padded # Now, all samples have had 5 µs of padding added to the front and back

time_steps = normalised_times_padded.shape[1]

# Define constants and extract needed data
bins = 100
electron_size = 58.5
areas = np.array(arr['area'])
print(max(areas)/electron_size)

# create a histogram of area distributions for gate, tritium and cathode data.
# first, boolean masks to filter gate, tritium and cathode data from main Awkward Array.
gate_events = arr[(arr['label'] == 1)].area / electron_size
tritium_events = arr[(arr['label'] == 2)].area / electron_size
cathode_events = arr[(arr['label'] == 0)].area / electron_size

# Create histograms without plotting
gate_hist = Hist(hist.axis.Regular(bins, 0, max(areas)/electron_size*1.01), label='S2 Gate Area Distribution')
gate_hist.fill(gate_events)

tritium_hist = Hist(hist.axis.Regular(bins, 0, max(areas)/electron_size*1.01), label='S2 Gate Tritium Distribution')
tritium_hist.fill(tritium_events)

cathode_hist = Hist(hist.axis.Regular(bins, 0, max(areas)/electron_size*1.01), label='S2 Gate Cathode Distribution')
cathode_hist.fill(cathode_events)

bin_edges = gate_hist.axes[0].edges
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Retrieve and adjust counts for each histogram to normalise counts across event types.
gate_counts = gate_hist.view() / bin_centers
tritium_counts = tritium_hist.view() / bin_centers
cathode_counts = cathode_hist.view() / bin_centers

print(cathode_hist.view().sum())

# Calculate the target flat spectrum as the average of the three histograms
gate_data = gate_hist.view(flow=False)
tritium_data = tritium_hist.view(flow=False)
cathode_data = cathode_hist.view(flow=False)

print(gate_counts.sum())
print(tritium_counts.sum())
print(cathode_counts.sum())

average_counts = np.mean([gate_counts.sum(), tritium_counts.sum(), cathode_counts.sum()])

# Calculate weights and reweighted data
gate_weights = []
gate_reweighted = []
for datapoint in range(len(gate_counts)):
    gate_reweighted.append(gate_counts[datapoint] * (1 / (gate_counts[datapoint] / gate_counts.sum())) if gate_counts[datapoint] != 0 else 0)
    gate_weights.append(1 / (gate_counts[datapoint] / gate_counts.sum()) if gate_counts[datapoint] != 0 else 0)

tritium_weights = []
tritium_reweighted = []
for datapoint in range(len(tritium_counts)):
    tritium_reweighted.append(tritium_counts[datapoint] * (1 / (tritium_counts[datapoint] / tritium_counts.sum())) if tritium_counts[datapoint] != 0 else 0)
    tritium_weights.append(1 / (tritium_counts[datapoint] / tritium_counts.sum()) if tritium_counts[datapoint] != 0 else 0)

cathode_weights = []
cathode_reweighted = []
for datapoint in range(len(cathode_counts)):
    cathode_reweighted.append(cathode_counts[datapoint] * (1 / (cathode_counts[datapoint] / cathode_counts.sum())) if cathode_counts[datapoint] != 0 else 0)
    cathode_weights.append(1 / (cathode_counts[datapoint] / cathode_counts.sum()) if cathode_counts[datapoint] != 0 else 0)

gate_reweighted = np.array(gate_reweighted)
tritium_reweighted = np.array(tritium_reweighted)
cathode_reweighted = np.array(cathode_reweighted)

# Creating the Weight Array to Feed into CNN
gate_weights = np.array(gate_weights)
gate_weights = np.where(np.isinf(gate_weights), 0, gate_weights)

tritium_weights = np.array(tritium_weights)
tritium_weights = np.where(np.isinf(tritium_weights), 0, tritium_weights)

cathode_weights = np.array(cathode_weights)
cathode_weights = np.where(np.isinf(cathode_weights), 0, cathode_weights)

print(f'these are the {gate_weights.size}')
print(gate_data.sum())
print(tritium_data.sum())
print(cathode_data.sum())

# Function to assign weights based on bin counts
def subdataset_total_weights(dataset_weights, n_data_per_bin):
    n_data_per_bin = np.array(n_data_per_bin, dtype=int)
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

# Creating and Populating the Weight Column
weight_column_4_mainarray = np.zeros(len(arr))

gate_event_counter = 0
cathode_event_counter = 0
tritium_event_counter = 0

# Assign weights to each training example based on its class
for i in range(len(arr)):
    if arr['label'][i] == 0:  # Cathode
        weight_column_4_mainarray[i] = c_weights[cathode_event_counter]
        cathode_event_counter += 1
    elif arr['label'][i] == 1:  # Gate
        weight_column_4_mainarray[i] = g_weights[gate_event_counter]
        gate_event_counter += 1
    else:  # Tritium (as we have already verified there are no None or NaN entries)
        weight_column_4_mainarray[i] = t_weights[tritium_event_counter]
        tritium_event_counter += 1

arr['weights'] = weight_column_4_mainarray

# Normalising the weights to have a mean of 1
num_events = len(arr['weights'])
total_weight = ak.sum(arr['weights'])
mean_weight = total_weight / num_events
weights_mean_one = arr['weights'] / mean_weight  # rescale all weights to have a mean of 1
arr = ak.with_field(arr, weights_mean_one, 'weights_normalised') # duplicate arr to now include the normalised weights
print(arr['weights_normalised'])

weights_np = ak.to_numpy(arr['weights'])
normalised_weights_np = ak.to_numpy(arr['weights_normalised'])

# An issue arises here initially as arr['weights'] is an awkward array. Keras only recognises and deals with a NumPy array therefore conversion is neccessary
# Another issue also arises in that the test and train data do not have asscoated weights as the weights column was initialised after the split was made
# 'arr' is the original dataset
normalised_area = ak.to_numpy(arr['area'] / electron_size)  # converting 'area' to detected electrons by dividing by 58.5

labels = arr['label']

radii = ak.to_numpy(arr['r'])

# TO ALTER BETWEEN WEIGHTS AND NORMLAISED WEIGHTS> CHANGE WEIGHTS_NP VARIABLE ACCORDINGLY> DEFAULT: NORMALISED WEIGHTS
X_train, X_test, y_train, y_test, area_train, area_test, weights_train, weights_test, \
normalised_times_train, normalised_times_test, times_us_train, times_us_test, \
normalised_samples_train, normalised_samples_test, r_train, r_test = train_test_split(
    X, labels, normalised_area, normalised_weights_np, normalised_times, time_us, normalised_samples, radii, 
    test_size=0.25, random_state=42
)

y_train = np.array(y_train) # this is neccessary as train_test_split often returns lists instead of ndarrays but Keras.model.fit requires the functionality of ndarrays
y_test = np.array(y_test)
print("Shape of y_train:", y_train.shape)

y_test_np = ak.to_numpy(y_test)
area_test_np = ak.to_numpy(area_test)

X_train_padded = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) # adding a channels dimension (greyscale of 1) to enable seamless input into CNN
X_test_padded = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convolutional Neural Network
seed_value = 42 # ensures reproducibility 
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# STEP 1: LEARNING RATE OPTIMIZATION
print("\n==== LEARNING RATE OPTIMIZATION ====")
# Define ranges for learning rates
min_lr = 1e-4
max_lr = 1e-1

# Generate random learning rates in a log-uniform distribution
num_trials = 40  # Number of random learning rates to test
random_learning_rates = np.random.uniform(np.log10(min_lr), np.log10(max_lr), num_trials)
random_learning_rates = 10 ** random_learning_rates  # Convert back to linear scale

# Placeholder for storing results
learning_rate_accuracies = []

# Loop through each random learning rate
for lr in random_learning_rates:
    print(f"Testing learning rate: {lr:.5e}")
    
    # Define the model
    model = Sequential([
        Conv1D(filters=32, kernel_size=100, strides=1, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model with the current learning rate
    optimizer = legacy.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])
    
    # Evaluate on the test set
    _, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save the accuracy for this learning rate
    learning_rate_accuracies.append((lr, test_accuracy))

# Sort the results by accuracy
learning_rate_accuracies = sorted(learning_rate_accuracies, key=lambda x: x[1], reverse=True)

# Print the best learning rate
best_lr, best_accuracy = learning_rate_accuracies[0]
print(f"\nBest Learning Rate: {best_lr:.5e}, Best Test Accuracy: {best_accuracy:.4f}")

# Plot learning rate vs. accuracy
plt.figure(figsize=(10, 6))
lrs, accuracies = zip(*learning_rate_accuracies)
plt.scatter(lrs, accuracies, marker='o', color='black', label="Accuracy")
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.legend()
plt.savefig(f'{base_dir}Figures/lr_optimization.png')
plt.show()

# Use the best learning rate
optimizer = legacy.Adam(learning_rate=best_lr)


# STEP 2: KERNEL SIZE RESEARCH
print("\n==== KERNEL SIZE RESEARCH ====")
# Define kernel sizes (combined from all ranges)
kernel_sizes_10us_to_1us = [1000, 800, 600, 400, 200, 100]  # [10 µs, 8 µs, ..., 1 µs]
kernel_sizes_1us_to_100ns = list(range(100, 9, -10))        # [100, 90, ..., 10]
kernel_sizes_100ns_to_20ns = list(range(10, 1, -1))         # [10, 9, ..., 2]
combined_kernel_sizes = kernel_sizes_10us_to_1us + kernel_sizes_1us_to_100ns + kernel_sizes_100ns_to_20ns

# Placeholder for accuracies and errors
combined_accuracies = []
combined_errors = []

# Number of runs per kernel size
num_runs = 3

# Run the model multiple times for each kernel size
for kernel_size in combined_kernel_sizes:
    accuracies = []  # Store accuracies for each run
    
    print(f"Training model with kernel size: {kernel_size} samples")
    for _ in range(num_runs):
        # Define the CNN model with default parameters
        model = Sequential([
            Conv1D(filters=32, kernel_size=kernel_size, strides=1, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
            MaxPooling1D(pool_size=1),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        # Compile the model with best learning rate
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model and get the accuracy
        history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        val_accuracy = history.history['val_accuracy'][-1]
        accuracies.append(val_accuracy)
    
    # Calculate the mean and standard deviation for the kernel size
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)  # Standard deviation
    sem = std_dev / np.sqrt(num_runs)  # Standard error of the mean
    
    # Store the mean accuracy and error
    combined_accuracies.append(mean_accuracy)
    combined_errors.append(sem)
    
    print(f"Kernel Size (samples): {kernel_size}, Mean Accuracy: {mean_accuracy:.4f}, SEM: {sem:.4f}")

# Convert kernel sizes to nanoseconds for plotting
combined_kernel_sizes_ns = [size * 10 for size in combined_kernel_sizes]  # Convert samples to nanoseconds

# Plot the data with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(np.log10(combined_kernel_sizes_ns), combined_accuracies, yerr=combined_errors, fmt='o', label='Validation Accuracy', capsize=5)

# Add labels and title
plt.xlabel("Log(Kernel Size in ns)")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs Kernel Size with Error Bars (Logarithmic Scale)")
plt.legend()
plt.savefig(f'{base_dir}Figures/kernel_size_accuracy.png')
plt.show()

# Find the best kernel size from our research
best_kernel_size_index = np.argmax(combined_accuracies)
best_kernel_size = combined_kernel_sizes[best_kernel_size_index]
print(f"Best kernel size identified: {best_kernel_size} samples ({best_kernel_size * 10} ns)")

# Placeholder for accuracies (one list per event type)
cathode_accuracies = []
tritium_accuracies = []
gate_accuracies = []

# Run the model multiple times for each kernel size and record class-specific accuracy
for kernel_size in combined_kernel_sizes:
    cathode_run_accuracies = []  # Store accuracies for cathode
    tritium_run_accuracies = []  # Store accuracies for tritium
    gate_run_accuracies = []     # Store accuracies for gate
    
    print(f"Training model with kernel size: {kernel_size} samples")
    for _ in range(num_runs):
        # Define the CNN model
        model = Sequential([
            Conv1D(filters=32, kernel_size=kernel_size, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
            MaxPooling1D(pool_size=1),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        
        # Get predictions for validation set
        val_split = int(0.2 * len(X_train_padded))  # Assuming 20% validation split
        X_val, y_val = X_train_padded[-val_split:], y_train[-val_split:]
        y_pred = np.argmax(model.predict(X_val), axis=1)
        
        # Calculate accuracies for each event type
        cathode_accuracy = np.mean(y_pred[y_val == 0] == 0)  # True positives for cathode
        tritium_accuracy = np.mean(y_pred[y_val == 2] == 2)  # True positives for tritium
        gate_accuracy = np.mean(y_pred[y_val == 1] == 1)     # True positives for gate
        
        # Store accuracies
        cathode_run_accuracies.append(cathode_accuracy)
        tritium_run_accuracies.append(tritium_accuracy)
        gate_run_accuracies.append(gate_accuracy)
    
    # Calculate mean accuracy across runs
    cathode_accuracies.append(np.mean(cathode_run_accuracies))
    tritium_accuracies.append(np.mean(tritium_run_accuracies))
    gate_accuracies.append(np.mean(gate_run_accuracies))
    
    print(f"Kernel Size (samples): {kernel_size}")
    print(f"  Cathode Mean Accuracy: {cathode_accuracies[-1]:.4f}")
    print(f"  Tritium Mean Accuracy: {tritium_accuracies[-1]:.4f}")
    print(f"  Gate Mean Accuracy: {gate_accuracies[-1]:.4f}")

# Plot the data for each class
plt.figure(figsize=(12, 8))
plt.plot(np.log10(combined_kernel_sizes_ns), cathode_accuracies, marker='o', label='Cathode Accuracy', color='blue')
plt.plot(np.log10(combined_kernel_sizes_ns), tritium_accuracies, marker='o', label='Tritium Accuracy', color='green')
plt.plot(np.log10(combined_kernel_sizes_ns), gate_accuracies, marker='o', label='Gate Accuracy', color='orange')

# Add labels and title
plt.xlabel("Log(Kernel Size in ns)")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs Kernel Size for Each Event Type (Logarithmic Scale)")
plt.legend()
plt.savefig(f'{base_dir}Figures/kernel_size_by_class.png')
plt.show()

# Placeholder for accuracies
training_accuracies = []
test_accuracies = []

# Run the model for each kernel size and compare training vs test accuracy
for kernel_size in combined_kernel_sizes:
    print(f"Training model with kernel size: {kernel_size} samples")
    
    # Define the CNN model
    model = Sequential([
        Conv1D(filters=32, kernel_size=kernel_size, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model with best learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    
    # Record training accuracy (last epoch)
    training_accuracy = history.history['accuracy'][-1]
    training_accuracies.append(training_accuracy)
    
    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    test_accuracies.append(test_accuracy)
    
    print(f"Kernel Size (samples): {kernel_size}")
    print(f"  Training Accuracy: {training_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

# Plot training vs test accuracy
plt.figure(figsize=(12, 8))
plt.plot(np.log10(combined_kernel_sizes_ns), training_accuracies, marker='o', label='Training Accuracy', color='blue')
plt.plot(np.log10(combined_kernel_sizes_ns), test_accuracies, marker='o', label='Test Accuracy', color='orange')

# Add labels and title
plt.xlabel("Log(Kernel Size in ns)")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy vs Kernel Size (Logarithmic Scale)")
plt.legend()
plt.savefig(f'{base_dir}Figures/training_vs_test.png')
plt.show()


# STEP 3: RANDOM SEARCH/BAYESIAN OPTIMIZATION
print("\n==== RANDOM SEARCH OPTIMIZATION ====")
# Random Search Optimisation
# Define ranges for hyperparameters
batch_size_range = [16, 32, 64, 218]
stride_range = [1, 2, 3, 5, 10, 20]
num_filters_range = [16, 32, 64, 128]
pool_size_range = [2, 3, 4, 5]

# Number of random samples to try
num_samples = 40

# Store results
random_search_results = []

for _ in range(num_samples):
    # Randomly sample hyperparameters
    batch_size = random.choice(batch_size_range)
    stride = random.choice(stride_range)
    filters = random.choice(num_filters_range)
    pool_size = random.choice(pool_size_range)
    
    print(f"Testing: Batch Size={batch_size}, Stride={stride}, Filters={filters}, Pool Size={pool_size}")
    
    # Define the model
    model = Sequential([
        Conv1D(filters=filters, kernel_size=best_kernel_size, strides=stride, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model with the best learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_padded, y_train, epochs=5, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[early_stopping])
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    # Append results
    random_search_results.append({
        'batch_size': batch_size,
        'stride': stride,
        'filters': filters,
        'pool_size': pool_size,
        'accuracy': test_accuracy
    })
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Sort results by accuracy
random_search_results = sorted(random_search_results, key=lambda x: x['accuracy'], reverse=True)

# Display top-performing hyperparameter combinations
print("\nTop 5 hyperparameter combinations from Random Search:")
for i, result in enumerate(random_search_results[:5]):
    print(f"{i+1}. Batch Size={result['batch_size']}, Stride={result['stride']}, Filters={result['filters']}, Pool Size={result['pool_size']}, Accuracy={result['accuracy']:.4f}")

# Visualize the accuracy of random samples
accuracies = [res['accuracy'] for res in random_search_results]
plt.figure(figsize=(10, 6))
plt.plot(range(len(accuracies)), accuracies, marker='o')
plt.xlabel("Random Sample Index")
plt.ylabel("Test Accuracy")
plt.title("Random Search: Test Accuracy vs Sample Index")
plt.savefig(f'{base_dir}Figures/random_search_results.png')
plt.show()

print("\n==== BAYESIAN OPTIMIZATION ====")
# Bayesian Optimisation
# Define hyperparameter space
space = [
    Integer(16, 512, name='batch_size'),                         # Batch size
    Integer(1, 50, name='stride'),                               # Stride
    Integer(16, 128, name='num_filters'),                       # Number of filters
    Integer(2, 5, name='pool_size')                             # Pool size
]

# Function to evaluate model performance with given hyperparameters 
@use_named_args(space)
def objective(batch_size, stride, num_filters, pool_size):
    print(f"Testing: Batch Size={batch_size}, Stride={stride}, Filters={num_filters}, Pool Size={pool_size}")
    
    # Define the model
    model = Sequential([
        Conv1D(filters=num_filters, kernel_size=best_kernel_size, strides=stride, activation='relu', input_shape=(X_train_padded.shape[1], 1)),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model with the best learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_padded, y_train, epochs=5, batch_size=batch_size, validation_split=0.2, verbose=0)
    
    # Evaluate the model
    _, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Minimise negative accuracy (maximise accuracy)
    return -test_accuracy

# Run Bayesian optimisation
results = gp_minimize(objective, space, n_calls=35, random_state=42)

# Extract the best result
best_hyperparams = results.x
best_score = -results.fun

print("\nBest Hyperparameters from Bayesian Optimization:")
print(f"Batch Size: {best_hyperparams[0]}")
print(f"Stride: {best_hyperparams[1]}")
print(f"Number of Filters: {best_hyperparams[2]}")
print(f"Pool Size: {best_hyperparams[3]}")
print(f"Best Test Accuracy: {best_score:.4f}")

# Visualize the convergence of Bayesian optimisation
plt.figure(figsize=(10, 6))
plt.plot(-np.array(results.func_vals), marker='o')
plt.xlabel("Iteration")
plt.ylabel("Test Accuracy")
plt.title("Bayesian Optimisation: Test Accuracy vs Iteration")
plt.savefig(f'{base_dir}Figures/bayesian_optimization.png')
plt.show()

# Use the best parameters from optimization
best_batch_size = best_hyperparams[0]
best_stride = best_hyperparams[1]
best_filters = best_hyperparams[2]
best_pool_size = best_hyperparams[3]

print(f"Using best parameters - Learning Rate: {best_lr:.5e}, Batch Size: {best_batch_size}, Stride: {best_stride}, " +
      f"Filters: {best_filters}, Pool Size: {best_pool_size}, Kernel Size: {best_kernel_size} samples")


# STEP 4: TRIMMING RESEARCH
print("\n==== TRIMMING RESEARCH ====")
# Placeholder for accuracies
front_trim_accuracies = []

# Loop to incrementally remove values from the front
for front_trim in range(0, X_train_padded.shape[1], 50):  # Remove in steps of 50
    print(f"Trimming {front_trim} values from the front")
    
    # Trim the front of the waveform
    X_train_trimmed = X_train_padded[:, front_trim:]
    X_test_trimmed = X_test_padded[:, front_trim:]
    
    # Define the CNN model with best parameters
    model = Sequential([
        Conv1D(filters=best_filters, kernel_size=best_kernel_size, strides=best_stride, activation='relu', input_shape=(X_train_trimmed.shape[1], 1)),
        MaxPooling1D(pool_size=best_pool_size),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model with best batch size
    history = model.fit(X_train_trimmed, y_train, epochs=5, batch_size=best_batch_size, validation_split=0.2, verbose=0)
    
    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test_trimmed, y_test, verbose=0)
    front_trim_accuracies.append(test_accuracy)
    
    print(f"Front Trim: {front_trim}, Test Accuracy: {test_accuracy:.4f}")
    
    # Stop if accuracy decreases by more than 0.01
    if len(front_trim_accuracies) > 1 and (front_trim_accuracies[-1] < front_trim_accuracies[-2] - 0.01):
        print("Accuracy decreased by more than 0.01. Stopping.")
        break

# Plot front trim results
plt.figure(figsize=(12, 8))
plt.plot(range(0, len(front_trim_accuracies) * 50, 50), front_trim_accuracies, marker='o', label='Test Accuracy')
plt.xlabel("Values Removed from Front")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Front-End Trimming")
plt.legend()
plt.savefig(f'{base_dir}Figures/front_trim.png')
plt.show()

# Placeholder for accuracies
back_trim_accuracies = []

# Loop to incrementally remove values from the back
for back_trim in range(0, X_train_padded.shape[1], 50):  # Remove in steps of 50
    print(f"Trimming {back_trim} values from the back")
    
    # Trim the back of the waveform
    X_train_trimmed = X_train_padded[:, :-back_trim] if back_trim > 0 else X_train_padded
    X_test_trimmed = X_test_padded[:, :-back_trim] if back_trim > 0 else X_test_padded
    
    # Define the CNN model
    model = Sequential([
        Conv1D(filters=best_filters, kernel_size=best_kernel_size, strides=best_stride, activation='relu', input_shape=(X_train_trimmed.shape[1], 1)),
        MaxPooling1D(pool_size=best_pool_size),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_trimmed, y_train, epochs=5, batch_size=best_batch_size, validation_split=0.2, verbose=0)
    
    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test_trimmed, y_test, verbose=0)
    back_trim_accuracies.append(test_accuracy)
    
    print(f"Back Trim: {back_trim}, Test Accuracy: {test_accuracy:.4f}")
    
    # Stop if accuracy starts decreasing
    if len(back_trim_accuracies) > 1 and test_accuracy < back_trim_accuracies[-2] - 0.01:
        print("Accuracy decreased by more than 0.01. Stopping.")
        break

# Plot back trim results
plt.figure(figsize=(12, 8))
plt.plot(range(0, len(back_trim_accuracies) * 50, 50), back_trim_accuracies, marker='o', label='Test Accuracy')
plt.xlabel("Values Removed from Back")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Back-End Trimming")
plt.legend()
plt.savefig(f'{base_dir}Figures/back_trim.png')
plt.show()