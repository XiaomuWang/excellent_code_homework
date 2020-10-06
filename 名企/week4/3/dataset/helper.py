import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob


def gen_batch_function(data_folder, image_shape):
    """
    This function is a Generator function.
    Generator Functions are used in training to efficienty pass data to the graph inside a session
    :data_folder: Folder Containing all test and train data ((160, 576,3))
    :image_shape: Shape Of the input image
    """
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))  # First we load all the image names

    label_paths = {  # Then we take the names of labels files
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    # In the orginal data set background is red
    background_color = np.array([255, 0, 0])
    random.shuffle(image_paths)  # Shuflling the data set

    def get_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))  # First we load all the image names

        label_paths = {  # Then we take the names of labels files
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        background_color = np.array([255, 0, 0])  # In the orginal data set background is red

        random.shuffle(image_paths)  # Shuflling the data set

        for batch_i in range(0, len(image_paths),
                             batch_size):  # Onece we go in this loop this generate on batch of trainig data
            images = []
            mask_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                mask_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file),
                                            image_shape)  # (160, 576,3) image reduced to single colour
                # here the image label means a mask
                mask_image = scipy.misc.imresize(scipy.misc.imread(mask_image_file), image_shape)  # (160, 576, 3)

                mask_bg = np.all(mask_image == background_color,
                                 axis=2)  # we can first extract the backgroud red pixels
                mask_bg = mask_bg.reshape(*mask_bg.shape, 1)  # reshaping this to (160,576,1)
                mask_image = np.concatenate((mask_bg, np.invert(mask_bg)),
                                            axis=2)  # now there are two types of pixels in the label (foreground(road) and background)
                # two channel mask image - each pixel can belongs to two classes
                images.append(image)  # Trainig images
                mask_images.append(mask_image)  # Training Image labels

            yield np.array(images), np.array(mask_images)  # yield function is to generate a new batch in each iteration

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
