import matplotlib
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
######################################################################################################
# join a directory and list of filenames

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

# plot images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

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

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
# print confusion matrix
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,
                          y_pred=cls_pred)
    print("Confusion matrix:")

    print(cm)

    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))

# plot example errors
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != cls_test)

    image_paths = np.array(image_paths_test)[incorrect]

    images = load_images(image_paths=image_paths[0:9])

    cls_pred = cls_pred[incorrect]

    cls_true = cls_test[incorrect]

    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# calculate the predicted classes fo the entire test-set
def example_errors():
    generator_test.reset()

    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)

    cls_pred = np.argmax(y_pred, axis=1)

    plot_example_errors(cls_pred)

    print_confusion_matrix(cls_pred)

# load images
def load_images (image_paths):
    images = [plt.imread(path) for path in image_paths]

    return np.asarray(images)

# plot training history
def plot_training_history(history):
    # get the classification acc and loss for the training-set
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # get for the validation-set
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # plot the acc and loss for the traning-set
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss.')

    # plot for the test set
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss.')

    # plot title and legend
    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.show()
############################################################################
# Dataset: Knifey-Spoony

import knifey
knifey.maybe_download_and_extract()
knifey.copy_files()

train_dir = knifey.train_dir
test_dir = knifey.test_dir
##########################################################################
# Pre-trained model: VGG16
"""
The VGG16 model contains a convolutional part and a fully-connected (or dense) part
 which is used for classification.
If include_top=True then the whole VGG16 model is downloaded
which is about 528 MB.
If include_top=False then only the convolutional part of the VGG16 model is downloaded
which is just 57 MB.
"""
model = VGG16(include_top=True, weights='imagenet')

##########################################################################
# Input Pipeline
"""
The Keras API has its own way of creating the input pipeline for training a model using files.

First we need to know the shape of the tensors expected as input by the pre-trained VGG16 model. In this case it is images of shape 224 x 224 x 3.
"""
input_shape = model.layers[0].output_shape[1:3]
print(input_shape)

# input the data into the Neural Network, which will loop over the data for enternity
datagen_train = ImageDataGenerator(rescale=1./255,
                                   rotation_range=180,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=[0.9, 1.5],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 20

if True:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'

# create the actual data-generator
# generator_train = datagen_train.flow_from_directory(directory=train_dir,
#                                                     target_size=input_shape,
#                                                     batch_size=batch_size,
#                                                     shuffle=True,
#                                                     save_to_dir=save_to_dir)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

# generator_test = datagen_test.flow_from_directory(directory=test_dir,
#                                                   target_size=input_shape,
#                                                   batch_size=batch_size,
#                                                   shuffle=False)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

###############################################################################
# set step_set
steps_test = generator_test.n / batch_size
print("Steps test: {}".format(steps_test))

# get the file-paths for all the images
# image_paths_train = path_join(train_dir, generator_train.filenames)
# image_paths_test = path_join(test_dir, generator_test.filenames)

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

# get the class-numbers
cls_train = generator_train.classes
cls_test = generator_test.classes

# get the class-names for the dataset
class_names = list(generator_train.class_indices.keys())
print("Class names: ")
print(class_names)

# get the number of classes
num_classes = generator_train.num_classes
print("Number of classes: {}".format(num_classes))

###############################################################################
# plot a few images to see if data is correct
images = load_images(image_paths=image_paths_train[10:19])
cls_true = cls_train[10:19]
plot_images(images=images, cls_true=cls_true, smooth=True)

#################################################################################
# Class weights
from sklearn.utils.class_weight import  compute_class_weight
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

print(class_weight)
print(class_names)

#################################################################################\
# Example predictions
# need to load, resize images and do the actual prediction and show the result
def predict(image_path):
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # plot the image
    plt.imshow(img_resized)
    plt.show()

    # convert the PIL image to a numpy-array with the proper shape
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # use the VGG16 model to make a prediction
    pred = model.predict(img_array)

    # decode the output
    pred_decoded = decode_predictions(pred)[0]

    # print the predictions
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))
######################################################################
predict(image_path='images/parrot_cropped1.jpg')

# predict(image_path=image_paths_train[0])
# predict(image_path=image_paths_train[1])
# predict(image_path=image_paths_test[1])
######################################################################
# Transfer learning
model.summary()

"""
the last Convolusional layer is called 'block5_pool'
==> its ouput will be re-routed to a new fully-connected Neural Network
"""
transfer_layer = model.get_layer('block5_pool')

print(transfer_layer.output)
#######################################################################
# take the part of the VGG16 from its input-layer to the output of the transfer-layer
# and call it as Convolutional model 'cause it consists of all the Convolutional layers
# from the VGG16
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

# build a new model
# start a new Keras Sequential mode
new_model = Sequential()

# add the convolutional part of VGG16
new_model.add(conv_model)

# flatten the output of the VGG because it's from a convolutional layer
new_model.add(Flatten())

new_model.add(Dense(1024, activation='relu'))

new_model.add(Dropout(0.5))

new_model.add(Dense(num_classes, activation='softmax'))

# optimize
optimizer = Adam(lr=1e-3)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']

# print whether a layer in the VGG16 model should be trained
def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

print_layer_trainable()
################################################################################
epochs = 5 # 20 is chosen because of resulting in 20 data-points (one for each epoch)
steps_per_epoch = 5 # batch_size = 20 ==> each epoch consists of 2000 random images
# in Transfer learning, be interested in the pre-trained VGG16
# ==> disable training for all its layers
# conv_model.trainable = False
#
# for layer in conv_model.layers:
#     layer.trainable = False
#
# print_layer_trainable()
#
# #########################################################################
# new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#

#
# history = new_model.fit_generator(generator=generator_train,
#                                   epochs=epochs,
#                                   steps_per_epoch=steps_per_epoch,
#                                   class_weight=class_weight,
#                                   validation_data=generator_test,
#                                   validation_steps=steps_test)
#
# plot_training_history(history)
#
# result = new_model.evaluate_generator(generator_test, steps=steps_test)
#
# print("Test-set classification accuracy: {0:.2%}".format(result[1]))
#
# example_errors()

#################################################################################3
# Fine-tuning
conv_model.trainable = True

# train the last two convolutional layers whose names contain 'block5' or 'block4'.
for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable

print_layer_trainable()
optimizer_fine = Adam(lr=1e-7)

new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)


plot_training_history (history)

result = new_model.evaluate_generator(generator_test, steps=steps_test)

print("Test-set classification accuracy: {0:.2%}".format(result[1]))

example_errors()