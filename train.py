from function import *
import pathlib
import sys
import numpy as np
from PIL import Image
import skimage
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, TimeDistributed, GRU, \
    BatchNormalization, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import random

random.seed(1)

################################################## Data preparation ###################################################
print('------------------------------------------------------------------------')
print(f'Preparing data for training ...')
data_dir = pathlib.Path("../DATASETS/Project_6")
image_count = len(list(data_dir.glob('**/*.tif')))
print(f'Total number of images: {image_count}')

try:
    color = sys.argv[1]
except IndexError:
    color = 'rgb'
print(f"Color space : {color}")

try:
    model_type = sys.argv[2]
except IndexError:
    model_type = 'CNN'
print(f"Model type : {model_type}")

# Load dataset
X, Y = load_dataset(data_dir, color)

"""
# Consider only 100 images for simplicity (only to test the code)
idx = random.sample(range(0, 374), 100)
X = X[idx]
Y = Y[idx]
"""

# Normalize dataset (if necessary)
X = normalize_dataset(X, color)
# Split dataset [Training (80% images) and Test(20% images) set]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)
# Patch division
X_train_patches, Y_train_patches = patch_division(X_train, Y_train, size=36, stride=32)
X_test_patches, Y_test_patches = patch_division(X_test, Y_test, size=32, stride=28)
# Reshape according to the model
X_train, Y_train = reshape(X_train_patches, Y_train_patches)

############################################### Data Augmentation #####################################################
print('------------------------------------------------------------------------')
print('Start: Data Augmentation')
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
Xtrain_augmented = datagen.flow(X_train, Y_train, batch_size=32, shuffle=True)
train_crops = crop_generator(Xtrain_augmented, crop_length=32, nchannel=X_train.shape[-1])
print('Completed: Data Augmentation')

batch_x, batch_y = next(train_crops)
############################################## Model fitting ####################################################

# Configuration:
batch_size = 32
no_epochs = 20

# Callbacks:
pat = 5  # no of epochs with no improvement after which the training will stop
early_stopping = EarlyStopping(monitor='accuracy', patience=pat, verbose=1)
model_checkpoint = ModelCheckpoint(f'model{model_type}_{color}.h5', monitor='accuracy', verbose=1, save_best_only=True)

model = ModelSelection(model_type, input_shape=batch_x[1].shape)
print('------------------------------------------------------------------------')
print(f'Training the model with Augmented Data ...')

# Fit data to model
history = model.fit(train_crops, steps_per_epoch=X_train.shape[0] // batch_size, epochs=no_epochs,
                    verbose=1, callbacks=[early_stopping, model_checkpoint])
# steps_per_epoch = |training_size|// batch_size

############################################ Model Evaluation #######################################################
print('------------------------------------------------------------------------')
print(f'Model evaluation')
class_list = ['CLL', 'FL', 'MCL']
# Reshape according to the model
X_test_eval, Y_test_eval = reshape(X_test_patches, Y_test_patches)
# Loss & Accuracy
score = model.evaluate(X_test_eval, Y_test_eval, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
# Majority voting
X_test, Y_test = reshape(X_test_patches, Y_test_patches, mode='test')
cm_prediction, cm_winner = majority_voting(model, X_test, Y_test)
# Confusion matrix
cm_prediction = np.asarray(cm_prediction)
cm_prediction = cm_prediction.reshape(cm_prediction.shape[0] * cm_prediction.shape[1])
cm_patches = confusion_matrix(Y_test_eval, cm_prediction)
cm_img = confusion_matrix(Y_test[:, 0], cm_winner)
print('------------------------------------------------------------------------')
print(f'Print confusion matrix for patches.')
plot_confusion_matrix(cm_patches, classes=class_list, title='Confusion Matrix for patches')
print(f'Print Confusion matrix for images.')
plot_confusion_matrix(cm_img, classes=class_list, title='Confusion Matrix for images', cmap=plt.cm.Reds)
# Precision, Recall, F-score
print('------------------------------------------------------------------------')
report = classification_report(Y_test[:,0], cm_winner, target_names=class_list)
print(f'Classification report:')
print(report)

