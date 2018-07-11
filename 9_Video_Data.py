"""
INTRO
- how to use your own images
- use Knifey-Spoony data-set: contains knives, spoons, forks
- classes: knifey, spoony and forky
"""
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

import inception

import prettytensor as pt

import knifey
from knifey import num_classes
################################################
data_dir = knifey.data_dir
knifey.maybe_download_and_extract()
dataset = knifey.load()

"""
You can use your own images instead of loading the knifey-spoony data-set. You have to create a DataSet-object from the dataset.py module. The best way is to use the load_cached()-wrapper-function which automatically saves a cache-file with the lists of image-files, so you make sure that the ordering is consistent with the transfer-values created below.

The images must be organized in sub-directories for each of the classes. See the documentation in the dataset.py module for more details.
"""
# This is the code you would run to load your own image-files.
# It has been commented out so it won't run now.

# from dataset import load_cached
# dataset = load_cached(cache_path='my_dataset_cache.pkl', in_dir='my_images/')
# num_classes = dataset.num_classes

# training and test-set
class_names = dataset.class_names
print(class_names)

#gett the training-set
#return the file-paths of img, cls_train: class numbers as integers, labels_train: one-hot encoded array
image_paths_train, cls_train, labels_train = dataset.get_training_set()

print(image_paths_train[1])

#get the test-set
image_paths_test, cls_test, labels_test = dataset.get_test_set()

print(image_paths_test[1])

print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            cls_true_name = class_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
########################################################
# function for loading img

from matplotlib.image import imread

def load_images(image_paths):
    images = [imread(path) for path in image_paths]

    # convert to a numpy array and return it
    return np.asarray(images)

#########################################################
# check the data
#images = load_images(image_paths=image_paths_test[0:9])
#cls_true = cls_test[0:9]
#plot_images(images=images, cls_true=cls_true, smooth=True)
#########################################################

# load the inception model
model = inception.Inception()

# calculate Transfer-values
from inception import transfer_values_cache

# set the file-paths for the caches of training-set and test-set
file_path_cache_train = os.path.join(data_dir, 'inception-knifey-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-knifey-test.pkl')

print("Processing Inception transfer-values for training-images ...")

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                              image_paths=image_paths_test,
                                              model=model)

transfer_values_train.shape


transfer_values_test.shape

############################################################
# plot transfer-values

def plot_transfer_values(i):
    print("Input image:")

    image = imread(image_paths_test[i])
    plt.imshow(image, interpolation='spline16')
    plt.show()

    print("Transfer-values for the image using Inception model: ")

    img = transfer_values_test[i]
    img = img.reshape((32,64))

    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

#plot_transfer_values(1)

##########################################################
# Analysis transfer values using PCA
# use PCA to reduce the array-lengths of transfer-values from 2048 to 2

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

transfer_values = transfer_values_train[0:3000]
cls = cls_train[0:3000]

print(transfer_values.shape)

# use PCA to reduce array-lengths
transfer_values_reduced = pca.fit_transform(transfer_values)
print(transfer_values_reduced.shape)
##############################################################
#plot the reduced transfer-values
def plot_scatter(values, cls):
    #create a color-map with a different color for each class
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    idx = np.random.permutation(len(values))

    #get the color for each sample
    colors = cmap[cls[idx]]

    x = values[idx, 0]
    y = values[idx, 1]

    plt.scatter(x, y, color=colors, alpha=0.5)
    plt.show()

#plot_scatter(transfer_values_reduced, cls=cls)
################################################################
# create another neural network
# input: transfer-values
# output: predicted classes

transfer_len = model.transfer_len

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural network

x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=2014, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)

# optimization method
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step)

# classification accuracy
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##############################################################################
# Tensorflow run

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch():
    num_images = len(transfer_values_train)

    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        x_batch, y_true_batch = random_batch()


        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}


        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)


        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)

    idx = np.flatnonzero(incorrect)

    # Number of images to select, max 9.
    n = min(len(idx), 9)

    # Randomize and select n indices.
    idx = np.random.choice(idx,
                           size=n,
                           replace=False)

    # Get the predicted classes for those images.
    cls_pred = cls_pred[idx]

    # Get the true classes for those images.
    cls_true = cls_test[idx]


    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)

    # Plot the images.
    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)
######################################################################
# plot confusion matrix
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,
                          y_pred=cls_pred)

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
#######################################################################
batch_size = 256


def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)


        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)

def classification_accuracy(correct):
    return correct.mean(), correct.sum()
#########################################################################
#show accuracy
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

##########################################################################
#Result

print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=True)

optimize(num_iterations=5000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)