"""
A method called "Feature maximization" for visually analysing the inner-workings of a NN
The idea:
    Generate an image that maximizes individual features
    The image is initialized with a little random noise and then gradually changed
    using the gradient of the gien feature
"""
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import inception
###################################################
inception.maybe_download()

# Names of convolutional layers

def get_conv_layer_names():
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']

    model.close()

    return names
conv_names = get_conv_layer_names()
# print(conv_names)
# print(conv_names[:5])
##################################################
# finding the input image
def optimize_image(conv_id=None, feature=0,
                   num_iterations=30, show_progress=True):
    """
    find an image that maximizes the feature given by the conv_id and feature number
    :param conv_id: integer
    :param feature: index into the layer for the feature to maximize
    :param num_iterations:
    :param show_progress:
    """
    model = inception.Inception()

    resized_image = model.resized_image
    y_pred = model.y_pred

    if conv_id is None:
        # if we want to maximize a feature on the last layer
        # then we use the fully_connected layer
        loss = model.y_logits[0, feature]
    else:
        conv_name = conv_names[conv_id]
        tensor = model.graph.get_tensor_by_name(conv_name + ":0")

        # print("## Minh ##", tensor.shape)
        with model.graph.as_default():
            loss = tf.reduce_mean(tensor[:,:,:,feature])
            # print("## Minh 1 ##", tensor[:,:,:,feature].shape)

    gradient = tf.gradients(loss, resized_image)

    session = tf.Session(graph=model.graph)

    # generate a random image
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    for i in range(num_iterations):
        feed_dict = {model.tensor_name_resized_image: image}

        # calculate the predicted calss-scores, gradient and the loss-value
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # Squeeze the dimensionality for the gradient-array
        grad = np.array(grad).squeeze()
        # print("Minh ", grad)
        step_size = 1.0 / (grad.std() + 1e-8)

        # update the image
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print("Iteration: ", i)

            pred = np.squeeze(pred)

            pred_cls = np.argmax(pred)

            cls_name = model.name_lookup.cls_to_name(pred_cls,
                                                     only_first_name=True)
            cls_score = pred[pred_cls]

            msg = "Predicted clss-name: {0} (#{1}), score: {2:>7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))

            msg = "Gradient min: {0:>9.6%}, max: {1:>9.6%}, step size: {2:>9.2%}"
            print(msg.format(grad.min(), grad.max(), step_size))

            print("Loss: ", loss_value)

            print()

    model.close()

    return image.squeeze()

######################################################################
# plot image and noise
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

def plot_image(image):
    img_norm = normalize_image(image)

    plt.imshow(img_norm, interpolation='nearest')
    plt.show()

def plot_images(images, show_size = 100):
    # the show_size is the number of pixels to show for each image,
    # the max value is 299
    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        img = images[i, 0:show_size, 0:show_size, :]

        img_norm = normalize_image(img)

        ax.imshow(img_norm, interpolation=interpolation)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    """
    find 6 images that maximize the 6 first features in the layer
    """
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer: ", conv_names[conv_id])

    images = []

    for feature in range(1, 7):
        print("optimizing image for feature no.", feature)

        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        image = image.squeeze()
        images.append(image)
    images = np.array(images)
    print("######################################################")
    print(len(images))
    print(images)
    plot_images(images=images, show_size=show_size)
##########################################################################
# Results
image = optimize_image(conv_id=5, feature=2,
                       num_iterations=50, show_progress=True)
plot_image(image)

# optimize_images(conv_id=None, num_iterations=10)
# optimize_images(conv_id=70, num_iterations=10)