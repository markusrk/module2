# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import tflowtools as TFT
from copy import deepcopy
import matplotlib.pyplot as PLT


class case_holder:
    def __init__(self,dataset,tfrac=0.1,vfrac=0.1):
        self.train_features = []
        self.train_labels = []
        self.dataset = dataset
        org_length = len(dataset)
        for x in range(round(org_length*(1-tfrac-vfrac))):
            i = random.randint(0, len(self.dataset) - 1)
            popped = self.dataset.pop(i)
            self.train_features.append(deepcopy(popped[0]))
            self.train_labels.append(deepcopy(popped[1]))
        

        self.validation_features = []
        self.validation_labels = []
        
        for x in range(round(org_length*vfrac)):
            i = random.randint(0, len(self.dataset) - 1)
            popped = self.dataset.pop(i)
            self.validation_features.append(deepcopy(popped[0]))
            self.validation_labels.append(deepcopy(popped[1]))
        
        self.test_features = []
        self.test_labels = []
        for x in dataset:
            self.test_features.append(deepcopy(x[0]))
            self.test_labels.append(deepcopy(x[1]))

    def test_batch(self):
        return self.test_features,self.test_labels

    def validation_batch(self):
        return self.validation_features,self.validation_labels

    def train_next_batch(self,size=100):
        if size == 'full':
            return self.train_features, self.train_labels
        else:
            features = []
            labels = []
            for _ in range(size):
                i = random.randint(0, len(self.train_features) - 1)
                features.append(self.train_features[i])
                labels.append(self.train_labels[i])
            return features, labels

class MNIST_holder:


    def __init__(self,cases=1000):
        from tensorflow.examples.tutorials.mnist import input_data
        self.cases = cases
        self.mnist = input_data.read_data_sets("datasets/MNIST/", one_hot=True)
        self.features, self.labels = self.mnist.train.next_batch(self.cases)

    def train_full_batch(self):
        return self.features, self.labels

# different error functions
def CE(y,x):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=x)

def MSE(y,x):
    losses = tf.losses.mean_squared_error(y,x)
    reduced = tf.reduce_sum(losses)
    return reduced

error_funcs = {'CE':CE,'MSE':MSE}

def train(dims=[11,40,20,6],
          activation_func='tanh',
          softmax=True,
          cost_func=CE,
          lr= 0.5,
          vint = 10,
          bint = 10,
          acc_lim = 0.95,
          initial_weight_range=[-0.1,0.1],
          data_source='gen_wine_cases',
          case_count=1,
          vfrac=0.1,
          tfrac=0.1,
          mbs=1277,
          map_bs=20,
          epochs=10000,
          show_layers=None,
          dendogram_layers=None,
          show=True,
          map_layers = [1, 2]):


  #Training and validation accuracies
  train_acc= []
  val_acc = []
    
  # Import data
  dataset = getattr(TFT,data_source)(case_count=case_count)
  mnist = case_holder(dataset,tfrac=tfrac,vfrac=vfrac)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, dims[0]], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, dims[-1]], name='y-input')


  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    if initial_weight_range == "scaled":
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        initial = tf.Variable(tf.random_uniform(shape=shape,minval=initial_weight_range[0],maxval=[initial_weight_range[1]]))
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=getattr(tf.nn,activation_func)):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations



  previous_layer = x
  layers = []
  for i in range(1,len(dims)):
      layers.append(nn_layer(previous_layer,dims[i-1],dims[i],'layer'+str(i),act=tf.nn.relu))
      previous_layer = layers[-1]
  y = layers[-1]



  with tf.name_scope('error_func'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = error_funcs[cost_func](y_,y)
    #diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('netsaver_test'+ '/train', sess.graph)
  test_writer = tf.summary.FileWriter('netsaver_test' + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train == "train":
      xs, ys = mnist.train_next_batch(size=mbs)
    elif train == 'test':
      xs, ys = mnist.test_features, mnist.test_labels
    elif train == 'val':
        xs, ys = mnist.validation_features, mnist.validation_labels
    elif train == 'map':
        xs, ys = mnist.train_features[:map_bs], mnist.train_labels[:map_bs]
    else:
        raise Exception
    return {x: xs, y_: ys}

  for i in range(epochs):
    if i % bint == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict('train'))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))

      # Own code for pulling training accuracy to matplot graph
      train_acc.append([i,acc])

      if acc >= acc_lim: break
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict('train'),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict('train'))
        train_writer.add_summary(summary, i)



  train_writer.close()
  test_writer.close()


  # Display final test scores
  summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict('train'))
  print('Final training set accuracy: %s' % ( acc))


  # Code for displaying graphs

  if show:
    TFT.plot_training_history(train_acc,val_acc)

  if map_layers:
      for l in map_layers:
          _, activation = sess.run([merged,layers[l]],feed_dict=feed_dict('map'))
          TFT.display_matrix(activation, title="mapping of layer: "+ str(l))
      # for variable in tf.trainable_variables():
      #     if variable.name in real-map_layer:
              # _,values = sess.run([merged,variable],feed_dict=feed_dict('map'))
              # if 'weigths' in variable.name:
              #     TFT.display_matrix(values)
              # elif 'biases' in variable.name:
              #     TFT.display_vector(values)
              # else:
              #     raise Exception("wrong dimensionality on show layers")

  if show_layers:
      for variable in tf.trainable_variables():
          if variable.name in show_layers:
              _,values = sess.run([merged,variable],feed_dict=feed_dict('map'))
              if len(values.shape) == 2:
                  TFT.display_matrix(values, title="weights of: "+variable.name)
              elif len(values.shape) == 1:
                  TFT.display_vector(values, title="biases of: "+variable.name)
              else:
                  raise Exception("wrong dimensionality on map layers")

  if dendogram_layers:
      for l in dendogram_layers:
          _, activation = sess.run([merged,layers[l]],feed_dict=feed_dict('map'))
          y_s = []
          #for y in feed_dict('map')[x]:
          #    y_s.append(TFT.segmented_vector_string(y))
          TFT.dendrogram(activation,feed_dict('map')[y_], title="Dendogram, layer: "+str(l))



  PLT.show()



def main(dict):
  if tf.gfile.Exists('netsaver_test'):
    tf.gfile.DeleteRecursively('netsaver_test')
  tf.gfile.MakeDirs('netsaver_test')
  train(**dict)


if __name__ == '__main__':
    import json
    config_file = 'configs/'+ str(input('choose setup file: '))+'.txt'
    #config_file = 'configs/test.txt'
    f = open(config_file,'r')
    dict = json.load(f)
    main(dict)