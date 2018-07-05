#how to save and store the variables of NN
#during the optimization, we save the variables of the NN whenever its classification acc
#has improved on the validation-set.
#The optimization is aborted when there has been no improvement for 1000 iterations

#This strategy is call Early Stopping.
#Overfitting occurs when the NN is being trained for too long so it starts to learn the
#noise of the training-set.

import matplotlib
import  matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix

import time

from datetime import timedelta
import math
import os

#use prettyTensor
import prettytensor as pt

print('Tensorflow: ' + tf.__version__)

#load data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data_set/MNIST', one_hot=True)

print("Size of:")
print("- Training-set: \t\t{}".format(len(data.train.labels)))
print("- Test-set: \t\t{}".format(len(data.test.labels)))
print("- Validation-set: \t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_channels = 1
num_classes = 10
#############################################################################33

#plot img
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    #create fig
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        #plot img
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        #set xlabel
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#plot some img
#img = data.test.images[0:9]
#cls_true = data.test.cls[0:9]
#plot_images(img, cls_true)
#############################################################################

#Placeholder
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name = 'x')
#CNN needs 4-dim img [num_img, img_height, img_width, num_channels
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#############################################################################
#NN
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)
#get weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable
weight_conv1 = get_weights_variable(layer_name='layer_conv1')
weight_conv2 = get_weights_variable(layer_name='layer_conv2')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

#performance measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_true_cls, y_pred_cls)

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #tf.cast: F = 0, T=1

#################################################################################
#Saver
saver = tf.train.Saver()
save_dir = 'checkpoints/' #the directory to save and retrieve the data

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation') #the path for the check-point file
##############################################################################################
#tensorflow run

session = tf.Session()

#initialize var
def init_variables():
    session.run(tf.global_variables_initializer())

init_variables()
#######################################################
# perform optimization iterations
train_batch_size = 64
#best validatino acc seen so far
best_validation_acc = 0.0
#iteratino-number for last improvement
last_improvement = 0
#stop optimization
require_improvement = 2000
################################################################################

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global best_validation_acc
    global last_improvement

    start_time = time.time()

    for i in range (num_iterations):
        total_iterations += 1
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict= feed_dict_train)

        if (total_iterations % 100 == 0) or (i == (num_iterations-1)):
            acc_train = session.run(acc, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()

            if acc_validation > best_validation_acc:
                best_validation_acc = acc_validation
                last_improvement = total_iterations
                saver.save(sess=session, save_path=save_path)

                #a string to be printed below, shows improvement found
                improved_str = '**************************************'
            else:
                improved_str = ''

            #status message for printing
            msg = "Iter: {0:>6}, Train-batch acc: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))
        #if no improvement found:
        if total_iterations - last_improvement > require_improvement:
            print('No improvement found in a while. Stop optimization.')
            break
    end_time = time.time()
    time_diff = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
###########################################################################3
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls

    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()
############################################################################
#Calculate classifications
batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

#calculate the predicted class for the test-set
def predict_cls_test():
    return predict_cls(images = data.test.images,
                       labels=data.test.labels,
                       cls_true=data.test.cls)

def predict_cls_validation():
    return predict_cls(images=data.validation.images,
                       labels=data.validation.labels,
                       cls_true=data.validation.cls)

#classification acc
def cls_accuracy(correct):
    #when summing a boolean array, false means 0, true means 1
    correct_sum = correct.sum()

    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)

# show the performance
def print_test_accuracy (show_example_errors=False,
                         show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)

    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1}/{2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#plot convolutional weights
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)

    print("Mean:{0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

##############################################################################
print_test_accuracy()

plot_conv_weights(weights=weight_conv1)


optimize(num_iterations=10000)


print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)


plot_conv_weights(weights=weight_conv1)
#################################################################################

# Initialize var again

init_variables()
print_test_accuracy()

plot_conv_weights(weights=weight_conv1)
#############################################################################

# Reload best variables

saver.restore(sess=session, save_path=save_path)

print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

plot_conv_weights(weights=weight_conv1)

session.close()

