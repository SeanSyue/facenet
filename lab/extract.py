"""
Extract image features from certain face image dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.data_flow_ops import FIFOQueue

import facenet
import util


def main(args):
    # ========================================================================================
    # Step 1: setup computational graph
    # ========================================================================================
    with tf.Graph().as_default() as g:
        image_paths_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='labels')
        batch_size_placeholder = tf.compat.v1.placeholder(tf.int32, name='batch_size')
        control_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='control')
        phase_train_placeholder = tf.compat.v1.placeholder(tf.bool, name='phase_train')

        eval_input_queue = FIFOQueue(capacity=2000000,
                                     dtypes=[tf.string, tf.int32, tf.int32],
                                     shapes=[(1,), (1,), (1,)],
                                     shared_name=None, name=None)

        image_size = (args.image_size, args.image_size)
        eval_enqueue_op = eval_input_queue.enqueue_many(
            [image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
        image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size,
                                                                 args.nrof_preprocess_threads,
                                                                 batch_size_placeholder)

    # ========================================================================================
    # Step 2: load images and initialize batches
    # ========================================================================================

    # Get the paths for the corresponding images
    print("Scanning images in dataset... ")
    image_paths = util.get_image_paths(args.dataset_root)

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0, len(image_paths)), 1)
    image_paths_array = np.expand_dims(np.array(image_paths), 1)
    control_array = np.zeros_like(labels_array, np.int32)

    # ========================================================================================
    # Step 3: inference model and save embeddings
    # ========================================================================================
    with tf.compat.v1.Session(graph=g) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Load the model
        input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
        facenet.load_model(args.model, input_map=input_map)

        # Get output tensor
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        # Run forward pass to calculate embeddings
        print(f'Running forward pass on {args.dataset_type} ... ')
        sess.run(eval_enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                                   control_placeholder: control_array})

        # Run forward pass to save embeddings
        for image_path in tqdm.tqdm(image_paths):
            feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.batch_size}
            emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
            out_path = util.get_output_path(args.feature_out_path, image_path, args.dataset_root)
            util.write_feat(out_path, emb, args.dataset_type)

        coord.request_stop()
        coord.join(threads)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_root', type=str,
                        help='Path to the data directory.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('feature_out_path', type=str,
                        help='Path to output features')
    parser.add_argument('dataset_type', type=str, choices=['IJBC', 'MEGA'],
                        help='Select IJBC or MegaFace dataset (for facescrub, choose `mega`)')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=1)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Input pipeline thread count', default=4)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
