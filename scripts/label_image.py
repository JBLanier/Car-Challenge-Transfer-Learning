# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import cv2

import player

import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    csv_path = "tf_files/CH_02_Prepared/all.csv"
    frames_dir = "tf_files/CH_02_Prepared/all"
    model_file = "tf_files/retrained_graph.pb"
    input_layer = "DecodeJpeg"
    output_layer = "final_result"
    video_writer = None
    out_file = None
    write_to_video = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", help="image directory to be processed")
    parser.add_argument("--csv", help="csv to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--out", help="optional mp4 file to output to")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.frames:
        frames_dir = args.frames
    if args.csv:
        csv_path = args.csv
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.out:
        out_file = args.out
        write_to_video = True

    if write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(out_file, fourcc, 20.0, (640, 480))

    csv = np.genfromtxt(csv_path, dtype=None, delimiter=',', names=True)
    graph = load_graph(model_file)
    train_writer = tf.summary.FileWriter('tf_files/predict',
                                         graph)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    avg_angle = 0

    with tf.Session(graph=graph) as sess:

        file_name_placeholder = tf.placeholder(dtype=tf.string)
        file_reader = tf.read_file(file_name_placeholder)
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')

        for row in csv:
            file_name = "{}/{}.jpg".format(frames_dir, row[0])
            # frame = imageio.imread(file_name)
            true_angle = row[1]

            start = time.time()
            frame = sess.run(image_reader, feed_dict={file_name_placeholder: file_name})

            predicted_angle = sess.run(output_operation.outputs[0],
                                       {input_operation.outputs[0]: frame})

            delta_time = time.time() - start
            wait_time = 0

            predicted_angle = np.squeeze(predicted_angle)

            avg_angle = 0.9 * avg_angle + 0.1 * predicted_angle

            if player.display_frame(frame,
                                    true_angle=true_angle,
                                    predicted_angle=predicted_angle,
                                    avg_angle=avg_angle,
                                    debug_info=str(row[0]),
                                    milliseconds_time_to_wait=1,
                                    video_writer=video_writer):
                break

    if write_to_video:
        video_writer.release()

    print("\n\nDone. Processed all frames.\n")

