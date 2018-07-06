#How to use several NN and average their outputs
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

import prettytensor as pt

#Load data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot = True)

print("Size of:")
print("- Training-set: \t\t{}".format(len(data.train.labels)))
print("- Test-set: \t\t{}".format(len(data.test.labels)))
print("- Validation-set: \t{}".format(len(data.validation.labels)))

#class numbers
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

#create rondom training-sets
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

print(combined_images.shape)
print(combined_labels.shape)

combine_size = len(combined_images)
print(combine_size)

train_size = int(0.8 * combine_size)
print(train_size)

validation_size = combine_size - train_size
print(validation_size)

#split combine data
def random_training_set():
    #create a randomized index into the full/combinded training-set
    idx = np.random.permutation(combine_size)

    #split the random index into training- and validation- sets
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    #select the img and labels for the new training-set
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    return x_train, y_train, x_validation, y_validation

#data dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_channels = 1
num_classes = 10

#plot img
def plot_images(images, cls_true, ensemble_cls_pred=None, best_cls_pred=None):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0 #cao len vi can viet nhieu data hon
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    #Voi moi subplot thi sao?
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            #plot img
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            #show true/predicted classes
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net:{2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])
            #show classes on x-axis
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]

plot_images(images, cls_true=cls_true)

#placeholder
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_clas = tf.argmax(y_true, dimension=1)

#############################################################################################
#Neural network
x_pretty = pt.wrap(x_image)

#create NN
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2). \
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

#Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

#performance measure
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_clas)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#saver
saver = tf.train.Saver(max_to_keep=100) #max_to_keep = so luong NN toi da
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

##################################################################################################
#tensorflow
session = tf.Session()

def ini_variables():
    session.run(tf.initialize_all_variables())

train_batch_size = 100
def random_batch(x_train, y_train):
    num_images = len(x_train)

    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]

    return x_batch, y_batch

def optimize(num_iterations, x_train, y_train):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch(x_train, y_train)
        feed_dict_train = {x:x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i%100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training batch Accuracy: {1:>6.1%}"
            print(msg.format(i+1, acc))
    end_time = time.time()
    time_diff = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

#################################################################################################
#Create ensemble of NN
num_networks = 5
num_iterations = 500

if True:
    #for each NN
    for i in range(num_networks):
        print("Neural Network: {0}".format(i))

        x_train, y_train, _, _ = random_training_set()
        session.run(tf.global_variables_initializer())

        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        saver.save(sess=session, save_path=get_save_path(i))

        print()

################################################################################################
#Calculating and predicting classifications
batch_size = 256

def predict_labels(images):
    num_images = len(images)
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :]}

        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    return pred_labels

def correct_prediction (images, labels, cls_true):
    pred_labels = predict_labels(images=images)
    cls_pred = np.argmax(pred_labels, axis=1)
    correct = (cls_true == cls_pred)

    return correct

def test_correct():
    return correct_prediction(images=data.test.images,
                              labels = data.test.labels,
                              cls_true=data.test.cls)

def validation_correct():
    return correct_prediction(images=data.validation.images,
                              labels=data.validation.labels,
                              cls_true=data.validation.cls)

##################################################################################################
#calculate the classification accuracy
def classification_accuracy(correct):
    return correct.mean()

def test_accuracy():
    correct = test_correct()

    return classification_accuracy(correct)

def validation_accuracy():
    correct = validation_correct()

    return classification_accuracy(correct)

#################################################################################################
def ensemble_predictions():
    pred_labels = []

    test_accuracies = []

    val_accuracies = []

    for i in range(num_networks):
        saver.restore(sess=session, save_path=get_save_path(i))

        test_acc = test_accuracy()

        test_accuracies.append(test_acc)

        val_acc = validation_accuracy()

        val_accuracies.append(val_acc)

        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        pred = predict_labels(images=data.test.images)

        pred_labels.append(pred)

    return np.array(pred_labels), \
            np.array(test_accuracies), \
            np.array(val_accuracies)


pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

pred_labels.shape

ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_pred_labels.shape

ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
ensemble_cls_pred.shape

ensemble_correct = (ensemble_cls_pred == data.test.cls)

ensemble_incorrect = np.logical_not(ensemble_correct)

test_accuracies

best_net = np.argmax(test_accuracies)
best_net

#the best NN's classification accuracy on the test-set
test_accuracies[best_net]

#predicted labels of the best NN
best_net_pred_labels = pred_labels[best_net, :, :]

#the predicted class number
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)

best_net_correct = (best_net_cls_pred == data.test.cls)

#boolean array whether each img is incorrectly classified
best_net_incorrect = np.logical_not(best_net_correct)
################################################################################################
#COmparision of ensemble vs the best single network

#the num of img in the test-set that were correctly classified by the ensemble
np.sum(ensemble_correct)

np.sum(best_net_correct)

ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
ensemble_better.sum()

best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)
best_net_better.sum()

###############################################################################################
#plot and prin comparision
def plot_images_comparison(idx):
    plot_images(images=data.test.images[idx, :],
                cls_true=data.test.cls[idx],
                ensemble_cls_pred=ensemble_cls_pred[idx],
                best_cls_pred=best_net_cls_pred[idx])

def print_labels (labels, idx, num=1):
    labels = labels[idx, :]
    labels = labels[0:num, :]

    labels_rounded = np.round(labels, 2)

    print(labels_rounded)
#print the predicted labels for the ensemble of NN
def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)

def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)

def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)

plot_images_comparison(idx=ensemble_better)

plot_images_comparison(idx=best_net_better)

