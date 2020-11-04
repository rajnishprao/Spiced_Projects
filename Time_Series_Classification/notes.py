'''
# found interesting dataset to classify time series data
# Human Activity Recognition (HAR): ts data from 3-axial accelerometer and gyroscope sensors (50 Hz sampling rate)
# sensors in smartphone worn on waist of test subjects as part of a research experiment
# data at: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

# short summary of expt/data
# 30 volunteers wearing devices, 70% randomly assigned to training, rest to test dataset
# sensory signals pre-processed by applying noise filters
# then sampled in fixed-width sliding windows of 2.56 sec & 50% overlap (128 readings/window)
# Sensor acceleration data has gravitational and body motion components
# these separated by Butterworth low-pass filter into body acceleration and gravity
# gravitational force assumed to have only low frequency components, therefore filter with 0.3 Hz cutoff was used
# for each window, a vector of features was obtained by calculating variables from time and freq domain
# (check features.info.txt for details)

# for each record provided:
# triaxial acceleration from accelerometer (total acceleration) and estimated body acceleration
# triaxial angular velocity from gyroscope
# a 561-feature vector with time and freq domain variables
# its activity label
# an identifier of subject who carried out the expt

# Dataset included these files:
# readme.txt - stuff from above
# features.info.txt - list of all features
# activity_labels.txt - links class label with their activity name
# train/test data + associated labels files

# following analysis template as per: https://github.com/jeandeducla/ML-Time-Series
# this repo uses different approaches to ts classification using ml tools
# in addition to testing different ml models (dl, nn, svm etc), two different types of inputs also tested
# these are raw ts data - or using features extracted from ts (statistical measures, freq domain features etc)
# HAR data for 6 different activities: walking, walking upstairs, walking downstairs, sitting, standing, lying
# each sample in dataet is a 2.56 s window sampled at 50 Hz
# which results in 6 x 128 readings per sample
# ie 3 accelerometer axes x, y, z & 3 gyroscope axes x, y z
# in this repo, we will only use the 3 accelerometer axes x, y, z
# lets try and plot a sample of each as in the repo info page

# there seems to be a better tutorial here:
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# a few more notes:
# raw data is not available but instead a pre-processed version of dataset
# pre-processed accelerometer and gyroscope using noise filters
# splitting data into fixed windows of 2.56 sec (128 data points) with 50% overlap
# feature engg was applied to the window data and a copy of the data with these engg features are available
# a no. of time and freq features commonly used in field of human activity recog was extracted for each window
# result was a 561 element vector of features
# data split 70/30 train/test ie 21 subjects for train & 9 for test
# expt results with SVM intended for use on smartphone (eg fixed point arithmetic) gave 89% accuracy on test set
# this was similar to an unmodified SVM implementation
# downloaded and saved files into HARDataset folder

# this tutorial is to develop a 1D CNN
'''

'''
CNNs were originally developed for image classification problems
where model learns an internal representation of a 2D input, in a processed
known as feature learning

same process can be applied to 1D data as in the HAR dataset
model learns to extract features from sequences of observations and how to
map the internal features to different activity types

benefit of CNN for sequence classification is it can learn from raw ts directly
ie deep domain knowledge not required to manually engineer input features
'''

'''
Load Data

3 main signal types in raw data: total acceleration, body acceleration and
body gyroscope - each of which has 3 axes of data. Therefore, there are a
total of 9 variables for each time step

further, each series of data is partitioned into overlapping windows of
2.56 sec of data or 128 time steps. These windows of data correspond to the
windows of engineered features (rows) in the previous section

this means that each row of data (2.56 sec) has 128 * 9 = 1152 data elements
(this is less than double the size of 561 element vectors in the previous
secion and it is likely that there is some redundant data - ???)

the signals are stored in Intertial Signals folder under train and test
subdirectories - each axis of each signal is stored in a separate file,
meaning that each of train and test datasets have 9 input files to load
and one output file to load. we batch the loading of these files into groups
given the consistent directory structures and file naming conventions
'''

# input data is in csv format (??) where columns are separated by whitespace (eh?)
# each of these files can be loaded as a numpy array - load_file() function below
# loads the dataset given the file path and returns the loaded data as a numpy array

from pandas import read_csv
from numpy import dstack

# laod a single file into a numpy array
def load_file(filepath):
    df = read_csv(filepath, header=None, delim_whitespace=True)
    return df.values

load_file('./HARDataset/train//Inertial Signals/body_acc_x_train.txt')


# we can then load all data for a given group (train or test) into a single
# 3D numpy array - where the dimensions of the array are [samples, time steps, features]
#
# to make this clearer; there are 128 time steps and 9 features, where the
# number of samples is the number of rows in any given raw signal data
#
# the load_group() function below implements this behaviour.
#
# the dstack() numpy function allows us to stack each of the loaded 3D arrays into
# a single 3D array where the variables are separated on the third dimension (features)

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
        # stack group so that features are the 3rd dimenion
        loaded = dstack(loaded)
        return loaded

load_group(train)
