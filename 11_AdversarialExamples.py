# Find a noise that makes the model mis-classify the image to our desired target_class
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import inception

print(tf.__version__)

#download and load the inception data
inception.maybe_download()
model = inception.Inception()

# In input and output for the inception model
resized_image = model.resized_image
#get a reference to the output of the softmax-classifier for the model
y_pred = model.y_pred

y_logits = model.y_logits

#hack the Inception model
# Set the graph for the Inception model as the default graph

with model.graph.as_default():
    # add a placeholder variable for the target class-number
    pl_cls_target = tf.placeholder(dtype=tf.int32)

    # add a new loss-function.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits,
                                                          labels=[pl_cls_target])

    # get the gradient for the loss - function with regard to the resize input image
    gradient = tf.gradients(loss, resized_image)
###################################################################
# Tensor flow Session
session = tf.Session(graph=model.graph)

# function for finding Adversary Noise
def find_adversary_noise(image_path, cls_target, noise_limit=3.0,
                         required_score=0.99, max_iterations=1000):
    """
    :param image_path:  the input-image (must be *.jpg
    :param cls_target: target class_numbeer (integer)
    :param noise_limit: limit for pixel-values in the noise

    """
    feed_dict = model._create_feed_dict(image_path=image_path)

    # calculate the predicted class-scores and the resized image
    pred, image = session.run([y_pred, resized_image],
                              feed_dict=feed_dict)

    # convert to one-dimensional array
    pred = np.squeeze(pred)

    # predict class-number
    cls_source = np.argmax(pred)

    # score for the predicted class
    score_source_org = pred.max()

    # names for the source and target classes
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)

    # Initialize the noise to zero
    noise = 0

    for i in range(max_iterations):
        print("iteration: ", i)

        noisy_image = image + noise

        # ensure the pixel-values of the noisy image ar between 0 and 255
        # like a real image.
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

        # create a feed-dict.
        feed_dict = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}

        # calculate the predicted class_scores
        pred, grad = session.run([y_pred, gradient],
                                 feed_dict=feed_dict)

        # convert the predicted class-scores to a one-dim array
        pred = np.squeeze(pred)

        # the scores for the source and target classes
        score_source = pred[cls_source]
        score_target = pred[cls_target]

        # squeeze the dimensionality for the gradient-array
        grad = np.array(grad).squeeze()

        # calculate the max of the absolute gradient values
        # this's used to calculate the step-size
        grad_absmax = np.abs(grad).max()

        # if the gradient is very small ==> use a lower limit
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # calculate the step-size for updating the image-noise
        # this step-size was found to give fast convergence
        step_size = 7 / grad_absmax

        # print the score etc. for the source - class
        msg = "Source score: {0:>7.2%}, class- number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        # print the score etc. for the target - class
        msg = "Target score: {0:>7.2%}, class- number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        # print statistics for the gradient
        msg = "Gradient min {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))

        print()

        # if the score for the target-class is not high enough
        if score_target < required_score:
            # update the image-noise by subtracting the gradient scaled by the step-size
            noise -= step_size * grad

            # ensure the noise is within the desired range
            # this avoids distorting the image too much
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            break
    return image.squeeze(), noisy_image.squeeze(), noise, \
           name_source, name_target, \
           score_source, score_source_org, score_target

# plot image and noise
def normalize_image(x):
    # get the min and max values for all pixels in the input
    x_min = x.min()
    x_max = x.max()

    # normalize
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

# plot the original img, noisy img, and the noise
def plot_images(image, noise, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    fig, axes = plt.subplots(1, 3, figsize=(10,10))

    #adjust vertical apacing
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(image / 255.0, interpolation=interpolation)
    msg = "Original Image: \n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    # plot the noisy image
    ax = axes.flat[1]
    ax.imshow(noisy_image / 255.0, interpolation=interpolation)
    msg = "Noisy Image: \n{0}, ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    # plot the noise
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel="Amplified NOise"
    ax.set_xlabel(xlabel)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# find and plot adversarial example
def adversarial_example(image_path, cls_target,
                        noise_limit, required_score):
    # find the adversarial noise
    image, noisy_image, noise, \
    name_source, name_target, \
    score_source, score_source_org, score_target = \
    find_adversary_noise(image_path=image_path,
                         cls_target=cls_target,
                         noise_limit=noise_limit,
                         required_score=required_score)

    # plot the img and the noise
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source,
                score_source_org=score_source_org,
                score_target=score_target)

    # print some statistics for the noise
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(),
                     noise.mean(), noise.std()))
###################################################################
# result
# image_path = "images/parrot_cropped1.jpg"
# image_path = "images/elon_musk.jpg"
image_path = "data/my_images/motor/motor2.jpg"
adversarial_example(image_path=image_path,
                    cls_target=200,
                    noise_limit=3.0,
                    required_score=0.99)

