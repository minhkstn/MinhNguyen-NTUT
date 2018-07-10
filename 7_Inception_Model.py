# How to use a pre-trained Deep NN called Inception v3 for image classification
# The inception v3 model takes weeks to train on a monster computer
# ==> will download the pre-trained Inception model and use it
# The Inception v3 model has:
#   25 mil parameters
#   5 bil multiply-ad operations
# for classifying a single image

# The model has 2 softmax-output.
#   one: during training
#   the other: classify images

# softmax-output should be called: classification score or ranks

# The Inception model works on Input images that are 299 x 299
# Img is resized automatically by the Inception model



import matplotlib

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import inception

print(tf.__version__)

inception.maybe_download()

# load the inception model
model = inception.Inception()

#display image

def classify(image_path):
    # classify
    pred = model.classify(image_path=image_path)
    # print the scores and names for the top-10 predictions
    print("\nPrint the scores:")
    model.print_scores(pred=pred, k=10, only_first_name=True)

    # display the img
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()





image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')

classify(image_path)

#classify(image_path="images/parrot.jpg")

#plot resized img
def plot_resized_image(image_path):
    resized_image = model.get_resized_image(image_path=image_path)
    plt.imshow(resized_image, interpolation='nearest')

    plt.show()

#plot_resized_image(image_path="images/parrot.jpg")

#cropped img
#classify(image_path="images/parrot_cropped1.jpg")
#classify(image_path="images/parrot_cropped3.jpg")

classify(image_path="images/minh_1.jpg")
classify(image_path="images/minh_2.jpg")
