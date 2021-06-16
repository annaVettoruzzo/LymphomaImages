import sys
import pathlib
from function import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import random
random.seed(1)

try:
    color = sys.argv[1]
except IndexError:
    color = 'rgb'
print(f"Color : {color}")

try:
    model_type = sys.argv[2]
except IndexError:
    model_type = 'CNN'
model_name = f'model{model_type}_{color}.h5'
print(f"Model name : {model_name}")

# DATA PREPARATION
print('------------------------------------------------------------------------')
data_dir = pathlib.Path("../DATASETS/Project_6")
image_count = len(list(data_dir.glob('**/*.tif')))
# Load dataset
X, Y = load_dataset(data_dir, color)
# Consider only 50 images for simplicity (only to test the code)
idx = random.sample(range(0, 373), 50)
X = X[idx]
Y = Y[idx]
# Plot a random image
idx = random.sample(range(0, 49), 1)
imageplotting(X[idx], color)

# Normalizing dataset
X = normalize_dataset(X, color)
# Split dataset [Training (80% images) and Test(20% images) set]
# maintaining the same proportion of elements from each class
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)
# Patch division
X_test_patches, Y_test_patches = patch_division(X_test, Y_test, size=32, stride=28)


# MODEL EVALUATION:
print('------------------------------------------------------------------------')
print(f'Load model: {model_name}')
model = keras.models.load_model(model_name)
model.summary()

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
report = classification_report(Y_test[:, 0], cm_winner, target_names=class_list)
print('Classification report:')
print(report)
