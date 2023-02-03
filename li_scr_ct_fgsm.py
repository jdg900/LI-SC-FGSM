# coding=utf-8
"""Implementation of MI-FGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.misc import imread, imsave
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

slim = tf.contrib.slim

tf.flags.DEFINE_float('beta', 1.5, 'the bound for variance tuning.')

tf.flags.DEFINE_float('std', 0.05, 'std for gaussian sampling.')

tf.flags.DEFINE_integer('batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_integer('num_ahead', 17, 'max iteration.')

tf.flags.DEFINE_integer('num_var', 5, 'max iteration')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_integer('patch_size', 200, 'mixed patch size')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './results',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS

seed=np.random.randint(0, 999)
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def grad_finish(x, one_hot, i, x_max, x_min, momentum, grad, variance):
    max_iter = FLAGS.num_ahead
    return tf.less(i, max_iter)

def variance_finish(x, one_hot, i, beta, grad):
    max_iter = FLAGS.num_var
    return tf.less(i, max_iter)

def batch_grad(x, one_hot, i, beta, grad):
    x_neighbor = x + tf.random.uniform(x.shape, minval=-beta, maxval=beta)

    x_neighbor_2 = 1 / 2 * input_mix_resize_uni(x_neighbor)
    x_neighbor_4 = 1 / 4 * input_mix_resize_uni(x_neighbor)
    x_neighbor_8 = 1 / 8 * input_mix_resize_uni(x_neighbor)
    x_neighbor_16 = 1 / 16 * input_mix_resize_uni(x_neighbor)

    x_res = tf.concat([x_neighbor, x_neighbor_2, x_neighbor_4, x_neighbor_8, x_neighbor_16], axis=0)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x_res), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
        #grad += tf.gradients(cross_entropy, x_neighbor)[0]
        grad += tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_res)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)
    i = tf.add(i, 1)
    return x, one_hot, i, beta, grad

def look_ahead(x, one_hot, i, x_max, x_min, momentum, grad, variance):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    num_classes = 1001

    x_nes = x + alpha * momentum

    x_nes_2 = 1 / 2 * input_mix_resize_uni(x_nes)
    x_nes_4 = 1 / 4 * input_mix_resize_uni(x_nes)
    x_nes_8 = 1 / 8 * input_mix_resize_uni(x_nes)
    x_nes_16 = 1 / 16 * input_mix_resize_uni(x_nes)

    x_batch = tf.concat([x_nes, x_nes_2, x_nes_4, x_nes_8, x_nes_16], axis=0)
    
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x_batch), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    new_grad = tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)

    v_iter = tf.constant(0)
    _, _, _, _, global_grad = tf.while_loop(variance_finish, batch_grad, [x, one_hot, v_iter, eps*FLAGS.beta, tf.zeros_like(new_grad)])
    
    current_grad = new_grad + variance
    noise = tf.nn.depthwise_conv2d(current_grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)

    noise = momentum + noise
    
    variance = global_grad / (1. * FLAGS.num_var)  - new_grad

    grad += noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, one_hot, i, x_max, x_min, noise, grad, variance

def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    beta = FLAGS.beta * eps
    std=FLAGS.std

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    pred = tf.argmax(end_points_v3['Predictions'], 1)

    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5, axis=0)
    #cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    #noise = tf.gradients(cross_entropy, x)[0]
    
    iter = tf.constant(0)
    _, _, _,  _, _, _, look_ahead_grad, _ = tf.while_loop(grad_finish, look_ahead, [x, one_hot, iter, x_max, x_min, tf.zeros_like(x), tf.zeros_like(x),tf.zeros_like(x)])
    #look_ahead_grad = look_ahead_grad / tf.reduce_mean(tf.abs(look_ahead_grad), [1, 2, 3], keep_dims=True)

    noise = look_ahead_grad
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)

    noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)

def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret
    # return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)

def input_mix_resize_uni(input_tensor):
    rnd = tf.random_uniform((), FLAGS.patch_size, FLAGS.image_width, dtype=tf.int32)
    resized = tf.image.resize_images(input_tensor, (rnd, rnd))

    start_x2 = tf.random_uniform((), 0, FLAGS.image_width - rnd, dtype=tf.int32)
    start_y2 = tf.random_uniform((), 0, FLAGS.image_height - rnd, dtype=tf.int32)
    
    patch_pad1 = tf.image.pad_to_bounding_box(resized, start_y2, start_x2, FLAGS.image_height, FLAGS.image_width)
    
    patch2 = tf.image.crop_to_bounding_box(input_tensor, start_y2, start_x2, rnd, rnd)
    patch_pad2 = tf.image.pad_to_bounding_box(patch2, start_y2, start_x2, FLAGS.image_height, FLAGS.image_width)

    trunc_img = (input_tensor - patch_pad2) 
    mix_img = trunc_img + patch_pad1

    return mix_img

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        #s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])

            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
