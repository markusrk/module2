# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
import numpy as np
import tflowtools as TFT

FLAGS = None

def get_next_cases(number,dataset):
    features = []
    labels = []
    for _ in range(number):
        i = random.randint(0,len(dataset)-1)
        features.append(dataset[i][0])
        labels.append(dataset[i][1])
    return features, labels


def gen_glass_cases():

    quality_mapping = {1:0,2:1,3:2,5:3,6:4,7:5}

    f = open('datasets/glass.txt','r')
    features, labels = ([],[])
    for line in f:
        split_line = list(map(float,line.split(",")))
        features.append(split_line[:len(split_line)-1])
        labels.append( quality_mapping[split_line[-1]])
    features = np.array(features)
    features_norm = features/features.max(0)
    output = []
    for i in range(len(features_norm)):
        output.append([features_norm[i].tolist(),TFT.int_to_one_hot(int(labels[i]),6)])
    return output

def main(layers = [9,27,6]):
  # Import data
  cases = gen_glass_cases()

  # Create the model
  input = tf.placeholder(tf.float32, [None, layers[0]])
  prev_layer = input
  output = None
  for i in range(1,len(layers)):
      with tf.name_scope("layer: "+str(i)):
          W = tf.Variable(tf.random_uniform([layers[i-1], layers[i]], minval=-.1, maxval=.1, dtype=tf.float32), trainable=True)
          b = tf.Variable(tf.zeros([1, layers[i]], dtype=tf.float32) + 0.1, name='b', trainable=True)
          output = tf.nn.relu(tf.matmul(prev_layer, W) + b,name="Layer"+str(i))
          prev_layer = output

  #W = tf.Variable(tf.random_uniform([9, 6], minval=-.1, maxval=.1, dtype=tf.float32), trainable=True)
  #b = tf.Variable(tf.zeros([6]))
  #output = tf.nn.relu( tf.matmul(input, W) + b)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 6])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(output)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'output', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
  train_step = tf.train.MomentumOptimizer(learning_rate=0.2,momentum=0.3).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(6000):
    batch_xs, batch_ys = get_next_cases(20,cases)
    sess.run(train_step, feed_dict={input: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_features, test_labels = get_next_cases(100,cases)
  print(sess.run(accuracy, feed_dict={input: test_features,
                                      y_: test_labels}))

if __name__ == '__main__':
    for _ in range(10):
      main(layers=[9,27,9,6])

