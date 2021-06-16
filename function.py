import numpy as np
from PIL import Image
import skimage
from skimage.color import rgb2lab, rgb2hed
from skimage.exposure import rescale_intensity
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import itertools


############################### DATA PREPROCESSING FUNCTIONS ###################################
# Function to CONVERT an image into Lab color space
def lab_converter(img):
    # Discard alpha channel which makes wheel round rather than square
    img = np.array(img.convert('RGB'))
    # Convert to Lab colorspace
    lab = rgb2lab(img)
    return lab


# Function to CONVERT an image into H&E color staining
def he_converter(img):
    img = np.array(img.convert('RGB'))
    hed = rgb2hed(img)
    return hed


# Function to LOAD dataset in the required color space
def load_dataset(data_dir, color):
    print('Start: Loading Dataset')
    class_list = ['CLL', 'FL', 'MCL']
    X = []
    Y = []
    for name in class_list:
        print('Extracting : ' + name + '/*.tif')
        for filename in data_dir.glob(name + '/*.tif'):
            im = Image.open(filename)
            if color == 'rgb':
                X.append(np.array(im))
            elif color == 'grayscale':
                X.append(np.array(im.convert('L')))
            elif color == 'lab':
                X.append(lab_converter(im))
            elif color == 'he':
                hed = rescale_intensity(he_converter(im), out_range=(0, 1))
                X.append(hed)
            elif color == 'h':
                h = he_converter(im)[:, :, 0]
                h = rescale_intensity(h, out_range=(0, 1))
                X.append(h)
            elif color == 'e':
                e = he_converter(im)[:, :, 1]
                e = rescale_intensity(e, out_range=(0, 1))
                X.append(e)

            Y.append(name)
    X = np.asarray(X)
    Y = np.asarray(Y)
    if len(X.shape) == 3:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    print("X shape: " + str(X.shape))
    print("Y shape: " + str(Y.shape))

    # To int [CLL -> 0, FL -> 1, MCL -> 2]
    target = np.zeros(Y.shape, dtype=int)
    target[Y == 'FL'] = 1
    target[Y == 'MCL'] = 2
    Y = target
    print('Completed: Loading Dataset')
    return X, Y


# Function to NORMALIZE the dataset, if necessary
def normalize_dataset(X, color):
    if color == 'rgb' or color == 'lab' or color == 'grayscale':
        X = X / 255.0
    return X

# Function to VISUALIZE images
def imageplotting(img, color):
    if color == 'grayscale':
        sampleX = img.reshape((img.shape[1], img.shape[2]))
        sampleX = Image.fromarray(sampleX, 'L')
    elif color == 'rgb':
        sampleX = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
        sampleX = Image.fromarray(sampleX)
    sampleX.show()


# EXTRACT patches from each image
def extract_patches(im, patch_size, step_size):
    patches = skimage.util.view_as_windows(im, patch_size,
                                           step_size)  # Windows are overlapping views of the input array,
    # with adjacent windows shifted by a single row or
    # column (or an index of a higher dimension)
    nR, nC, t, H, W, C = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W, C))
    patches = np.asarray(patches)  # dtype=np.float16)
    return patches


def patch_division(X, Y, size, stride):
    print('Start: Patch Division')
    X_patches = []
    Y_patches = []
    for i in range(X.shape[0]):
        label = Y[i]
        p = extract_patches(X[i], patch_size=(size, size, X.shape[3]), step_size=stride)
        X_patches.append(p)
        Y_patches.append([label for j in range(p.shape[0])])
    X_patches = np.asarray(X_patches)
    Y_patches = np.asarray(Y_patches)
    print('Completed: Patch Division')
    return X_patches, Y_patches


# RESHAPE dataset according to model/purpose of usage
def reshape(X, Y, mode='fit'):
    """
    if (type == 'CNN' or type == 'CNNRNN') and mode == 'fit':
        X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Y_reshaped = Y.reshape((Y.shape[0] * Y.shape[1]))
    elif (type == 'CNN' or type == 'CNNRNN') and mode == 'test':
        X_reshaped = X
        Y_reshaped = Y
    elif color == 'grayscale' or color == 'h' or color == 'e':  # implicit: model_type = 'RNN'
        if mode == 'fit':
            X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            Y_reshaped = Y.reshape((Y.shape[0] * Y.shape[1]))
        else:  # implicit: mode == 'test'
            X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
            Y_reshaped = Y
    elif type == 'RNN' and mode == 'fit':
        X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2] * X.shape[3], X.shape[4]))
        Y_reshaped = Y.reshape((Y.shape[0] * Y.shape[1]))
    else:  # implicit: type == 'RNN' and mode == 'test'
        X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2] * X.shape[3], X.shape[4]))
        Y_reshaped = Y
    """
    if mode == 'fit':
        X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Y_reshaped = Y.reshape((Y.shape[0] * Y.shape[1]))
    else:
        X_reshaped = X
        Y_reshaped = Y
    print("X reshaped: " + str(X_reshaped.shape))
    print("Y reshaped: " + str(Y_reshaped.shape))
    return X_reshaped, Y_reshaped


############################### DATA AUGMENTATION FUNCTIONS ####################################
# Function to random CROP an image
def random_crop(img, random_crop_size):
    # Following the idea in the paper of Janowczyk: randomly crop out smaller patches from the larger ones
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    #    if img.ndim == 2:
    #        return img[y:(y + dy), x:(x + dx)]
    return img[y:(y + dy), x:(x + dx), :]


# Function to create an image generator to which apply random crop
def crop_generator(batches, crop_length, nchannel):
    """Take as input a Keras ImageGen (Iterator) and generate random
     crops from the image batches generated by the original iterator.
     """
    while True:
        batch_x, batch_y = next(batches)
        #        if batch_x[0].ndim == 2:
        #            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length))
        #        else:
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, nchannel))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops, batch_y


############################### MODEL DEFINITION  #########################################
def ModelCNN(input_shape):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Dropout(0.2))

    # Passing it to Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2rd Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(3, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
    return model


def ModelRNN(input_shape):
    model = Sequential()

    if input_shape[-1] == 1:
        # One channel only -> Exploit spatial relation between x and y axis
        model.add(Reshape(input_shape[0:2], input_shape=input_shape))
    else:
        # Multiple channels -> Exploit relation between each entry (single pixel)
        model.add(Reshape(np.flip(input_shape), input_shape=input_shape))
        model.add(TimeDistributed(Flatten()))
        shape = model.layers[-1].output_shape
        model.add(Reshape(np.flip(shape[1:])))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, entry_size)
    model.add(GRU(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    return model


def ModelCNNRNN(input_shape):
    model = Sequential()

    # CNN Layers
    model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1),
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # Max Pooling with pool size and strides = 4 in order to maintain the same dimensions as in the CNN
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Dropout(0.2))

    # Reshape layers to perform the flatting of the pixels
    shape1 = model.layers[-1].output_shape
    model.add(Reshape(np.flip(shape1[1:])))
    model.add(TimeDistributed(Flatten()))
    shape2 = model.layers[-1].output_shape
    model.add(Reshape(np.flip(shape2[1:])))

    # RNN layers
    model.add(GRU(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(128, activation='relu'))
    model.add(Dropout(0.2))

    # Fully Connected Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    return model


def ModelSelection(model_type, input_shape):
    print('Start: Defining model ' + model_type)
    if model_type == 'CNN':
        model = ModelCNN(input_shape)
    elif model_type == 'RNN':
        model = ModelRNN(input_shape)
    elif model_type == 'CNNRNN':
        model = ModelCNNRNN(input_shape)
    print('Completed: Defining model' + model_type)
    return model


############################# MODEL EVALUATION ######################################################
def majority_voting(model, X_test, Y_test):
    test_accuracy = 0.0
    cm_prediction = []
    cm_winner = []
    for i in range(0, X_test.shape[0]):  # Loop for each image:
        prediction = model.predict(X_test[i])
        candidates = np.argmax(prediction, axis=1)
        cm_prediction.append(candidates)
        counts = np.bincount(candidates)  # count winners for each class
        winner = np.argmax(counts)  # majority voting: select the winner
        cm_winner.append(winner)
        if all(x == winner for x in Y_test[i]):  # ground truth check
            test_accuracy = test_accuracy + 1.0  # accuracy update
            if np.max(counts) < 2 / 3 * len(X_test[i]):
                print(f"WARNING: Image {i + 1} is correctly classified, but with less than 2/3 of patches being correctly classified.")
        else:
            print(f'Image {i + 1} is wrongly classified.')
    test_accuracy = test_accuracy / X_test.shape[0]
    print('------------------------------------------------------------------------')
    print(f'> Image classification accuracy in Test dataset: (majority-voting) {test_accuracy}')
    print('------------------------------------------------------------------------')
    return cm_prediction, cm_winner


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
