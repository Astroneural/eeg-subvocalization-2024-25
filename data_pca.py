import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import pathlib
from keras import utils, regularizers

import matplotlib.pyplot as plt
import pandas as pd

from keras import layers
from keras import models
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, Dropout,  MaxPooling2D, GlobalAveragePooling2D
import keras_tuner as kt

from scipy import signal
from scipy.signal import butter, lfilter, medfilt, welch
from skimage.restoration import denoise_wavelet
from sklearn.decomposition import FastICA

def convert(csv):
  return genfromtxt(csv, delimiter=',')

train_data = convert('/Users/shaum/eeg-stuffs/ut5_train.csv')
test_data = convert("/Users/shaum/eeg-stuffs/ut5_test.csv")
val_data = convert('/Users/shaum/eeg-stuffs/ut5_val.csv') # swapping these two lead to different results. 
# This way ~70% is achievable with val accuracies similar. Swapped test accuracy goes to ~75% but there is greater discrepancies.

# testing

'''import seaborn as sns

# Select the columns of interest
columns_of_interest = train_data[:, 2:7]

# Specify the range
start_index = 530 # A: 530-661, B: 662-793, C: 794-925
end_index = 926

# Create a DataFrame with the selected columns
df = pd.DataFrame(columns_of_interest[start_index:end_index+1])

# Rename the columns
df.columns = ['AF3', 'T7', 'Pz', 'T8', 'AF4']

# Plot the data using seaborn with solid lines
sns.lineplot(data=df, dashes=False)

# Add dashed vertical lines at x = 661 and x = 793
plt.axvline(x=(661-530), color='black', linestyle='--')
plt.axvline(x=(793-530), color='black', linestyle='--')

# Add labels "A", "B", and "C" above the graph
plt.text(64, df.max().max() + 20, "A", ha='center', fontsize=12)
plt.text(192, df.max().max() + 20, "B", ha='center', fontsize=12)
plt.text(320, df.max().max() + 20, "C", ha='center', fontsize=12)

# Remove the x and y axis labels
plt.xlabel('Time (128 Hz sampling frequency)')
plt.ylabel('Voltage fluctuation (µV)')

# Show the plot
plt.show()'''


'''train_data = convert('test_falguni_abc_3.csv')
test_data = convert("val_falguni_abc_4.csv")
val_data = convert('val_falguni_abc.csv')'''

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def convolve(data):
  filter = signal.firwin(400, [0.01, 0.06], pass_zero=False)
  return signal.convolve(data, filter, mode='same')

def medfit(data):
    med_filtered=signal.medfilt(data, kernel_size=3)
    return  med_filtered

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess(data):
    nsamples = data.shape[0]
    fs = 128.0
    lowcut = .3 # .4 is best I believe
    highcut = 60.0

    # Apply Butterworth bandpass filter to each channel
    for i in range(2, data.shape[1]):
        data[:, i][1:] = butter_bandpass_filter(data[:, i][1:], lowcut, highcut, fs, order=1) # 6 or 1 is best
    # Apply notch filter at 60 Hz to each channel
    for i in range(2, data.shape[1]):
        f0 = 60.0  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor
        b, a = signal.iirnotch(f0, Q, fs)
        data[:, i][1:] = lfilter(b, a, data[:, i][1:])
    
    
def preprocess_with_pca(data, n_components=5):
    nsamples = data.shape[0]
    fs = 128.0
    lowcut = .4 # .4 is best I believe
    highcut = 60.0

    # Apply Butterworth bandpass filter to each channel
    '''for i in range(2, data.shape[1]):
        data[:, i][1:] = butter_bandpass_filter(data[:, i][1:], lowcut, highcut, fs, order=1) # 6 or 1 is best'''

    # Apply median filter
    '''for i in range(2, data.shape[1]):
        data[:, i] = medfit(data[:, i])'''
    
    

    # Apply PCA
    pca_data = data[:, 2:]  # Exclude the first two columns (timestamps and labels)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)  # Standardize the data
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Concatenate the timestamps and labels with the PCA result
    preprocessed_data = np.hstack((data[:, :2], pca_result))

    return preprocessed_data


'''preprocessed_train_data_with_pca = preprocess_with_pca(train_data)
preprocessed_test_data_with_pca = preprocess_with_pca(test_data)
preprocessed_val_data_with_pca = preprocess_with_pca(val_data)'''

'''preprocessed_train_data_with_pca = train_data
preprocessed_test_data_with_pca = test_data
preprocessed_val_data_with_pca = val_data'''

preprocess(train_data)
preprocess(test_data)
preprocess(val_data)

preprocessed_train_data_with_pca = train_data #note: currently not actually with PCA
preprocessed_test_data_with_pca = test_data
preprocessed_val_data_with_pca = val_data


n_chan = 5

def convert_to_image_array_with_pca(data, n_components, image_length):
    line_index = 1
    current_word = 0
    current_image = np.zeros((1, n_components))
    image_directory = []
    answer_directory = []

    while line_index < data.shape[0]:
        current_line = data[line_index]

        if int(current_line[0]) == current_word:
            current_image = np.vstack((current_image, current_line[2:])) # adding the data to the current image
        else: # if the word has changed, save this image and make a new one
            current_word = int(current_line[0])
            current_image_trimmed = current_image[1:image_length+1]  
            image_directory.append(current_image_trimmed)
            answer_directory.append(current_line[1])
            current_image = np.zeros((1, n_components))

        line_index += 1

    image_directory = np.array(image_directory) # shape (n_images, image_length, n_components)
    answer_directory = np.array(answer_directory) # shape (n_images,)
    answer_directory = utils.to_categorical(answer_directory)

    return image_directory, answer_directory

# Define parameters
n_components = 5  # Number of components after PCA
image_length = 128  # sampling rate and duration (128 Hz, 1s samples)

# Convert preprocessed data with PCA to image arrays

train_data_imageDirectory, traindata_answerDirectory = convert_to_image_array_with_pca(preprocessed_train_data_with_pca, n_components, image_length)
test_data_imageDirectory, test_data_answerDirectory = convert_to_image_array_with_pca(preprocessed_test_data_with_pca, n_components, image_length)
val_data_imageDirectory, val_data_answerDirectory = convert_to_image_array_with_pca(preprocessed_val_data_with_pca, n_components, image_length)

x_train = train_data_imageDirectory
y_train = traindata_answerDirectory
x_test = test_data_imageDirectory
y_test = test_data_answerDirectory
x_val = val_data_imageDirectory
y_val = val_data_answerDirectory

np.random.seed(0)

def apply_ica(data, n_components):
    data = np.reshape(data, (data.shape[0], -1))  # Flatten channels and time_points
    ica = FastICA(n_components=n_components)
    ica_result = ica.fit_transform(data)

    return ica_result

n_components_ica = 5

ica_result_train = apply_ica(x_train, n_components_ica)
ica_result_test = apply_ica(x_test, n_components_ica)
ica_result_val = apply_ica(val_data_imageDirectory, n_components_ica)

ica_result_train_expanded = np.expand_dims(ica_result_train, axis=1)  # Expand dimensions along time_points axis
ica_result_test_expanded = np.expand_dims(ica_result_test, axis=1)
ica_result_val_expanded = np.expand_dims(ica_result_val, axis=1)

# Concatenate ICA features with original features
x_train_ica = np.concatenate((x_train, ica_result_train_expanded), axis=1)
x_test_ica = np.concatenate((x_test, ica_result_test_expanded), axis=1)
x_val_ica = np.concatenate((val_data_imageDirectory, ica_result_val_expanded), axis=1)

print("Shape of combined training data:", x_train_ica.shape)
print("Shape of combined testing data:", x_test_ica.shape)
print("Shape of combined validation data:", x_val_ica.shape)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""indices = range(3, len(x_val_ica), 3)
images = []

for index in indices:
    gray_image = Image.fromarray(x_val_ica[index])

    # Trim the image to (128, 5) by removing the last row
    trimmed_image = gray_image.crop((0, 0, gray_image.width, gray_image.height - 1))

    # Normalize the image to the range of 0 to 255
    normalized_image = np.array(trimmed_image)
    normalized_image = (normalized_image - np.min(normalized_image)) * (255 / np.ptp(normalized_image))
    
    images.append(normalized_image)

# Assuming all images are of the same dimension (128x5 in this case)
combined_image = np.zeros_like(images[0])

# Process each column individually
for col in range(combined_image.shape[1]):  # For each column
    # Extract the same column from all images and calculate the mean
    column_data = np.mean([image[:, col] for image in images], axis=0)
    combined_image[:, col] = column_data

# Re-normalize the combined image column-wise
min_vals = np.min(combined_image, axis=0)
max_vals = np.max(combined_image, axis=0)
for col in range(combined_image.shape[1]):  # For each column
    combined_image[:, col] = (combined_image[:, col] - min_vals[col]) * (255 / (max_vals[col] - min_vals[col]))

# Apply the Viridis color map to the combined, re-normalized image
colorized_image = cm.viridis(combined_image.astype(np.uint8))
colorized_image = (colorized_image * 255).astype(np.uint8)


# Display the colorized, combined image using matplotlib
plt.imshow(colorized_image)
plt.axis('off')  # Turn off axis labels
plt.savefig('combined_colorized_image_columnwise_a1.png', bbox_inches='tight', pad_inches=0)  # Save the image as PNG
plt.close()  # Close the plot to avoid displaying it"""



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

index = 107
gray_image = Image.fromarray(x_train_ica[index])

# Trim the image to (128, 5) by removing the last row
trimmed_image = gray_image.crop((0, 0, gray_image.width, gray_image.height - 1))

# Normalize the image to the range of 0 to 255
normalized_image = np.array(trimmed_image)
normalized_image = (normalized_image - np.min(normalized_image)) * (255 / np.ptp(normalized_image))

# Apply the Viridis color map
colorized_image = cm.viridis(normalized_image.astype(np.uint8))
colorized_image = (colorized_image * 255).astype(np.uint8)

print(y_train[index])

# Display the colorized image using matplotlib
plt.imshow(colorized_image)
plt.axis('off')  # Turn off axis labels
plt.savefig('train_colorized_image_c.png', bbox_inches='tight', pad_inches=0)  # Save the image as PNG
plt.close()  # Close the plot to avoid displaying it

'''Raw data analysis: a single plot of a chunk of data
Classification analysis: what's accuracy, analyse model, mne loss?, analyse filtering, what are the results (benefits) of filtering, analyse accuracy and results 
'''
