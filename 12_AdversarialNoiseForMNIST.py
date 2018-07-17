"""
Tutorial 11: the Adversarial noise was found through an optimization process
==> the noise was specialized for each image

This tutorial:
    Find a Adversarial noise that causes nearly all input imgs
to become mis-classified as a desired target class
    Make the NN immune to adversarial noise
"""
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import prettytensor as pt
##################################################################
# load data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_channels = 1
num_classes = 10

# plot images
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):  # for each image

        image = images[i].reshape(img_shape)

        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
#test
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
# plot_images(images=images, cls_true=cls_true)
####################################################################
# TensorFlow Graph
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Adversarial Noise
noise_limit = 0.35 # the noise will be between +- 0.35

"""
The optimizer will minimize 2 loss-measures:
    The normal loss-measure for the neural network
        ==> find the noise giving the best classification acc for the target-class
    The L2-loss-measure trying to keep the noise as low as possible
==> need a priority weight for these 2 loss-measures
"""
noise_l2_weight = 0.02  # close to zero usually works best

# When we create the new variable for the noise, we must inform TensorFlow
# which variable-collections that it belongs to, so we can later inform the
# two optimizers which variables to update.

# the name of new variable-collection
ADVERSARY_VARIABLES = 'adversary_variables'

# a list of collections
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]

# Create the new variable for the adversarial noise
# it will not trainable
# ==> will not be optimized along with the other variables of NN
# ==> we can create 2 separate optimization procedures
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))

x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

#############################################################################
# Convolutional NN
x_pretty = pt.wrap(x_noisy_image)

# add layers
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimizer for Normal training
print ([var.name for var in tf.trainable_variables()])

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Optimizer for Adversarial Noise
# get the list of variables being optimized in this procedure
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
print([var.name for var  in adversary_variables])

# Combine 2 loss-measures
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise

# optimizer for the advarsary noise
# give it a list of the variables that we want to update
# Note that the learning rage is much greater
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

#################################################################################
# Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#################################################################################
# TensorFlow Run
session = tf.Session()

session.run(tf.global_variables_initializer())

def init_noise():
    session.run(tf.variables_initializer([x_noise]))
init_noise()

# perform optimization iterations
train_batch_size = 64

def optimize(num_iterations, adversary_target_cls=None):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        if adversary_target_cls is not None:
            y_true_batch = np.zeros_like(y_true_batch)

            y_true_batch[:, adversary_target_cls] = 1.0

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        if adversary_target_cls is None:
            # Run the optimizer using this batch of training data
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            session.run(optimizer_adversary, feed_dict_train)

            # limit the adversarial noise
            session.run(x_noise_clip)
        # print status every 100 iterations
        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization iteration: {0:>6}, Training acc: {1:>6.1%}"
            print(msg.format(i, acc))
    end_time = time.time()
    time_diff = end_time - start_time

    print("Time Usage: " + str(timedelta(seconds=int(round(time_diff)))))

# get and plot the noise
def get_noise():
    noise = session.run(x_noise)

    return np.squeeze(noise)

def plot_noise():
    noise = get_noise()
    print("Noise:")
    print("- Min: ", noise.min())
    print("- Max: ", noise.max())
    print("- Std: ", noise.std())

    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)

# plot errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    noise = get_noise()

    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
###########################################################################
# show the performance
test_batch_size = 128

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = data.test.cls
    correct = (cls_true==cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test

    msg = "Acc on Test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred,correct=correct)
    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# optimize(num_iterations=1000)
#
# print_test_accuracy(show_example_errors=True)

init_noise()

optimize(num_iterations=1000, adversary_target_cls=3)
plot_noise()
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# find all noise
def find_all_noise (num_iterations=1000):
    all_noise = []

    for i in range(num_classes):
        print("Finding adversarial noise for target-class: ", i)

        init_noise()

        optimize(num_iterations=num_iterations,adversary_target_cls=i)

        noise = get_noise()

        all_noise.append(noise)

        print()

    return all_noise

# all_noise = find_all_noise(num_iterations=300)

# plot all noise
def plot_all_noise(all_noise):
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        noise = all_noise[i]

        ax.imshow(noise, cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)
        ax.set_xlabel(i)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# plot_all_noise(all_noise)

#############################################################################
# make a neural network immune to noise
# 1st: run the optimization to find the adversarial noise
# 2nd: run the normal optimization the make the neural network immune to that noise

def make_immune(target_cls, num_iterations_adversary=500,
                num_iterations_immune=1000):
    print("Target-class: ", target_cls)
    print("Finding adversarial noise ...")

    optimize(num_iterations=num_iterations_adversary,
             adversary_target_cls=target_cls)
    print()

    print_test_accuracy(show_confusion_matrix=False,
                        show_example_errors=False)
    print()
    print("Making the neural network immune to the noise ...")

    optimize(num_iterations=num_iterations_immune)

    print()

    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=True)

# for i in range(10):
#     make_immune(target_cls=i)
make_immune(target_cls=3)
print("######################################################################")