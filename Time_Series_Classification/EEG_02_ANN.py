# whats with tf and jupyter notebooks man??
# this works ok - but note the architechture - its an ANN
# why is the dense layer at the end have 1 neuron? something is weird about this whole input and output scheme
# increasing no. of epochs did not make a difference to the validation loss which is crap
# don't think this is a right CNN thing is a right thing to do for this kinda data
# maybe SVC is the way to go 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping


# import data
eeg_df = pd.read_csv('EEG_data.csv')
demo_df = pd.read_csv('demographic_info.csv')

# merging data frames on subject ID - but different nomenclature, so first rename in second df
demo_df.rename(columns={'subject ID':'SubjectID'}, inplace=True)
df = demo_df.merge(eeg_df, on='SubjectID')

# cleaning up column names which are in different formats
df.rename(columns={' age':'Age', ' ethnicity':'Ethnicity', ' gender':'Gender', 'user-definedlabeln':'Label'}, inplace=True)

# dropping subjectid, videoid and one of the labels predefinedlabel as they are not useful
df = df.drop(['SubjectID', 'VideoID', 'predefinedlabel'], axis=1)

# label is in floats - can change it to ints - looks better
df['Label'] = df['Label'].astype(np.int)

# encoding features
# replacing gender with 1 for M & 0 for F
df['Gender'] = df['Gender'].apply(lambda x:1 if x == 'M' else 0)

# getting dummies for ethnicity
ethnic_dummies = pd.get_dummies(df['Ethnicity']) #, drop_first=True)
df = pd.concat([df, ethnic_dummies], axis=1)
df = df.drop('Ethnicity', axis=1)

# scaling and splitting data
X = df.drop('Label', axis=1).copy()
y = df['Label'].copy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# first running CNN shown in notebook - if all is well, i'll try my own cnn
# what even is this format? not a sequential model but something else??
# seems like an older version of TF - don't use in future

inputs = tensorflow.keras.Input(shape=(X_train.shape[1]))
x = tensorflow.keras.layers.Dense(256, activation='relu')(inputs)
x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
outputs = tensorflow.keras.layers.Dense(1, activation='sigmoid')(x)

model = tensorflow.keras.Model(inputs, outputs)

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', tensorflow.keras.metrics.AUC(name='auc')])

batch_size = 32
epochs = 50

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size,
                    epochs=epochs, callbacks=[tensorflow.keras.callbacks.ReduceLROnPlateau()])

plt.figure(figsize=(16, 10))
plt.plot(range(epochs), history.history['loss'], label='Training Loss')
plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.legend()
plt.show()

# results
model.evaluate(X_test, y_test)

y_true = np.array(y_test)
y_pred = np.squeeze(model.predict(X_test)) # what on earth is this np.squeeze??
y_pred = np.array(y_pred >= 0.5, dtype=np.int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_pred, y_true))
