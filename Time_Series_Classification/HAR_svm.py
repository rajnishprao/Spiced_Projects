'''
Application of SVM to time series classification
Task: classify 6 different types of activities based on x, y, z accelerometer signals

Here, features are extracted first from raw signals and used as input for SVM

https://github.com/jeandeducla/ML-Time-Series/blob/master/SVM-Accelerometer.ipynb

not sure which of the two accelerometer data was used in above example so i will
stick to body_acc data
'''
import numpy as np
#import tensorflow as tf
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
#import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt


# body_acc raw signals
# x axis
X_train_x_raw = np.loadtxt('./HARDataset/train/Inertial Signals/body_acc_x_train.txt')
X_test_x_raw = np.loadtxt('./HARDataset/test/Inertial Signals/body_acc_x_test.txt')
# y axis
X_train_y_raw = np.loadtxt('./HARDataset/train/Inertial Signals/body_acc_y_train.txt')
X_test_y_raw = np.loadtxt('./HARDataset/test/Inertial Signals/body_acc_y_test.txt')
# z axis
X_train_z_raw = np.loadtxt('./HARDataset/train/Inertial Signals/body_acc_z_train.txt')
X_test_z_raw = np.loadtxt('./HARDataset/test/Inertial Signals/body_acc_z_test.txt')

print('X_train_x_raw shape:', X_train_x_raw.shape)
print('X_train_y_raw shape:', X_train_y_raw.shape)
print('X_train_z_raw shape:', X_train_z_raw.shape)
# cool so 7352 rows/epochs with 128 time points in each

print('X_test_x_raw shape:', X_test_x_raw.shape)
print('X_test_y_raw shape:', X_test_y_raw.shape)
print('X_test_z_raw shape:', X_test_z_raw.shape)
# and these have 2947 rows/epochs with 128 time points in each

# loading label vectors
y_train = np.loadtxt('./HARDataset/train/y_train.txt')
y_test = np.loadtxt('./HARDataset/test/y_test.txt')

print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)
# cool so numbers match no. of X_train and X_test samples

label_names = ['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying']


# Feature Extraction helper functions
# instead of raw signals, we are going to extract features from the signals
# following functions help build feature vectors out of raw signals
# they help extract statistical and geometric features from raw signals and
# jerk signals (acceleration first derivative)
# also we extract freq domain features from raw and jerk signals
# for each sample, we will extract the following features

# x, y, z raw signals: mean, max, min, std, skew, kurtosis, interquartile range,
# mean absolute deviation, area under curve, area under squared curve

# x, y, z jerk signals (first derivative): same list as above

# x, y, z raw signals Discrete Fourier Transform: same list as above, plus the
# following: weighted mean freq, 5 first DFT coefficients, 5 first local maxima
# of DFT coefficients and their corresponding frequencies

# x, y, z jerk signals Discrete Fourier Transform: same longer list as above

# x, y, z correlation coefficients

# quick note on the reshape function
# mean_example = np.mean(X_train_x_raw, axis=1) - this generates the mean of the
# 128 samples in each of the 7352 epochs/rows
# but when you check mean_example.shape: you get (7352,)
# which is bad - we need 7352 mean values in 1 column
# so you gotta reshape as follows
# mean_example = np.mean(X_train_x_raw, axis=1).reshape(-1, 1)
# mean_example.shape: now gives (7352, 1)

import scipy.stats as st
from scipy.fftpack import fft, fftfreq
from scipy.signal import argrelextrema
import operator

def stat_area_features(x, Te=1.0): # whats this Te??

    mean_ts = np.mean(x, axis=1).reshape(-1, 1) # mean
    max_ts = np.amax(x, axis=1).reshape(-1, 1) # max
    min_ts = np.amin(x, axis=1).reshape(-1, 1) # min
    std_ts = np.std(x, axis=1).reshape(-1, 1) # std
    skew_ts = st.skew(x, axis=1).reshape(-1, 1) # skew
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1, 1) # kurtosis
    iqr_ts = st.iqr(x, axis=1).reshape(-1, 1) # interquartile range
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1, 1)),
                        axis=1), axis=1).reshape(-1, 1) # median absolute deviationn - wetf
    area_ts = np.trapz(x, axis=1, dx=Te).reshape(-1, 1) # area under curve Te kinda makes sense if you look up documentation for np.trapz esp illustration image
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=Te).reshape(-1, 1) # area under curve ** 2

    return np.concatenate((mean_ts, max_ts, min_ts, std_ts, skew_ts, kurtosis_ts,
                           iqr_ts, mad_ts, area_ts, sq_area_ts), axis=1)


def frequency_domain_features(x, Te=1.0):

    # as DFT coefficients and their corresponding frequencies are symetrical arrays
    # with respect to the middle of the array, we need to know if the no. of readings
    # in x is even or odd to them split the arrays
    if x.shape[1]%2 == 0:
        N = int(x.shape[1]/2)
    else:
        N = int(x.shape[1]/2) - 1
    xf = np.repeat(fftfreq(x.shape[1], d=Te)[:N].reshape(1, -1), x.shape[0], axis=0) # frequencies - wtf is this?? gotta read fftfreq
    dft = np.abs(fft(x, axis=1))[:N] # DFT coefficients - gotta check these two. vaguely intuitive but not very sure

    # statistical and area features
    dft_features = stat_area_features(dft, Te=1.0)
    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1, 1)

    # first 5 DFT coefficients
    dft_first_coef = dft[:, :5]
    # first 5 local maxima of DFT coefficients and their corresponding frequencies
    dft_max_coef = np.zeros((x.shape[0], 5))
    dft_max_coef_f = np.zeros((x.shape[0], 5))
    for row in range(x-shape[0]):
        # find all local maximas indexes
        extrema_ind = argrelextrema(dft[row, :], np.greater, axis=0)
        # makes a list of tuples (DFT_i, f_i) of all local maxima
        # and keeps the 5 biggest - vaguely understand this - gotta get details into kopf
        extrema_row = sorted([(dft[row, :][j], xf[row,j]) for j in extrema_ind[0]],
                             key=operator.itemgetter(0), reverse=True)[:5]
        for i, ext in enumerate(extrema_row):
            dft_max_coef[row, i] = ext[0]
            dft_max_coef_f[row, i] = ext[1]

    return np.concatenate((dft_features, dft_weighted_mean_f, dft_first_coef,
                            dft_max_coef, dft_max_coef_f), axis=1)

def make_feature_vector(x, y, z, Te=1.0):

    # raw signals: stats and area features
    features_xt = stat_area_features(x, Te=Te)
    features_yt = stat_area_features(y, Te=Te)
    features_zt = stat_area_features(z, Te=Te)

    # jerk signals: stats and area features
    # ok what the figgity f is happening here?? i gotta look up details
    features_xt_jerk = stat_area_features((x[:, 1:]-x[:,:-1])/Te, Te=1/Te)
    features_yt_jerk = stat_area_features((y[:, 1:]-y[:,:-1])/Te, Te=1/Te)
    features_zt_jerk = stat_area_features((z[:, 1:]-z[:,:-1])/Te, Te=1/Te)

    # raw signals: freq domain features
    features_xf = frequency_domain_features(x, Te=1/Te)
    features_yf = frequency_domain_features(y, Te=1/Te)
    features_zf = frequency_domain_features(z, Te=1/Te)

    # jerk signals: freq domain features: again gotta look up what this selection means...
    features_xf_jerk = frequency_domain_features((x[:, 1:])/Te, Te=1/Te)
    features_yf_jerk = frequency_domain_features((y[:, 1:])/Te, Te=1/Te)
    features_zf_jerk = frequency_domain_features((z[:, 1:])/Te, Te=1/Te)

    # raw signals correlation coefficient between axis
    cor = np.empty((x.shape[0], 3))
    for row in range(x.shape[0]):
        xyz_matrix = np.concatenate((x[row,:].reshape(1, -1), y[row,:].reshape(1, -1),
                                     z[row,:].reshape(1, -1)), axis=0)
        cor[row, 0] = np.corrcoef(xyz_matrix)[0, 1]
        cor[row, 1] = np.corrcoef(xyz_matrix)[0, 2]
        cor[row, 2] = np.corrcoef(xyz_matrix)[1, 2]

    return np.concatenate((features_xt, features_yt, features_zt,
                            features_xt_jerk, features_yt_jerk, features_zt_jerk,
                            features_xf, features_yf, features_zf,
                            features_xf_jerk, features_yf_jerk, features_zf_jerk, cor),
                            axis=1)

# Te set to 1/50 here : does that mean 1 in 50 samples??
X_train = make_feature_vector(X_train_x_raw, X_train_y_raw, X_train_z_raw, Te=1/50)
X_test = make_feature_vector(X_test_x_raw, X_test_y_raw, X_test_z_raw, Te=1/50)

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

# scaling features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
