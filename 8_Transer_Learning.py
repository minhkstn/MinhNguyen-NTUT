# Tranfer Learning:
#   re-use the pre-trained Inception model
#   merely replace the layer that does the final classification

# use transfer-values as the input to another neural network
# train the second neural network
# ==> The Inception model: extract useful info from the images
# ==> Another neural network: classify the images

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

from tensorflow.contrib.keras.api.keras import activations

import inception
import prettytensor as pt

print (pt.__version__)

import cifar10

from cifar10 import num_classes

#############################################################
# Load data
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print (class_names)

# Load sets
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of: ")
print("- Training-set: \t{}".format(len(labels_train)))
print("- Test-set: \t{}".format(len(labels_test)))

############################################################################
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
        if i < len(images):
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
##############
images = images_test[0:9]
cls_true = cls_test[0:9]

#plot_images(images=images, cls_true=cls_true)

##############################################################
# Download the inception model
inception.maybe_download()

# Load the inception model
model = inception.Inception()

########################################################
# calculate transfer-values
from inception import transfer_values_cache

#set file-paths for the caches of the training- and test-set
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print("Processing Inception transfer-values for training-images ...")

# Scale img
images_scaled = images_train * 255.0

#load/ calculate + save transfer-values
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                             images=images_scaled,
                                             model=model)

print("Processing Inception transfer-values for test-images ...")

images_scaled = images_test * 255.0

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)
transfer_values_train.shape

transfer_values_test.shape

##################################################################################
# plot transfer values
def plot_transfer_values(i):
    print("Input image: ")

    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model: ")

    #transform the transfer-values into an image
    img = transfer_values_test[i]
    img = img.reshape((32,64))

    plt.imshow(img,interpolation='nearest', cmap='Reds')
    plt.show()

#################################################################################
# Analysis of transfer-values using PCA
# use principal component Analysis (PCA) from scikit-learn to reduce the array-lengths of
# the transfer-values from 2048 to 2, so they can be plotted

from sklearn.decomposition import PCA

# create a new PCA object and set the target array-length to 2
pca = PCA(n_components=2)

# limit the number of samples: 1000
transfer_values = transfer_values_train[0:1000]

# get the class numbers
cls = cls_train[0:1000]

# check the size
transfer_values.shape

# use PCA to reduce the array-length rom 2048 to 2
transfer_values_reduced = pca.fit_transform(transfer_values)
transfer_values_reduced.shape

# plot the reduced transfer-values
def plot_scatter(values, cls):
    #create a color-map with a different color for earch class
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    #get the color or each sample
    colors = cmap[cls]

    #extract the x- and y- values
    x = values[:, 0]
    y = values[:, 1]

    plt.scatter(x, y, color=colors)
    plt.show()

plot_scatter(transfer_values_reduced, cls)

###########################################################################################
# create another neural network
# input: transfer-values
# output: the predicted classes

transfer_len = model.transfer_len

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural network
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization method
# create a variable for kepping track of the number of optimization iterations
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

##########################################################################################
# classification accuracy
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#################################################################################################
# tensorflow run
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 36

def random_batch():
    num_images = len(transfer_values_train)

    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

#######################################################################################
# optimize
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    end_time = time.time()
    time_diff = end_time - start_time

    print("Time Usage: \t{}".format(time_diff))

#####################################################################################4
# show results
def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)

    images = images_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    n = min(9, len(images))

    # Plot the first n images.
    plot_images(images=images[0:n],
                cls_true=cls_true[0:n],
                cls_pred=cls_pred[0:n])

# plot confusion matrix
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,
                          y_pred=cls_pred)

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

# calculating classifications
batch_size = 100

def predict_cls(transfer_values, labels, cls_true):
    num_images = len(transfer_values)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_pred == cls_true)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test,
                       labels=labels_test,
                       cls_true=cls_test)

def classification_accuracy(correct):
    return correct.mean(), correct.sum()

########################################################################################
# print accuracy
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()

    acc, num_correct = classification_accuracy(correct)

    num_images = len(correct)

    msg = "Accuracy on Test-set: {0:.1%}, ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct= correct)
    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred)
#######################################################################################
# execute
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
optimize(num_iterations=10000)

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)