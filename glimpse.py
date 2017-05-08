from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, images_ph):
    '''

    :param config: Configuration
    :param images_ph: Batch Input to Core Network
    '''
    self.config = config
    self.dropout_keep_prob = tf.constant(0.5)
    self.images_ph = images_ph

    # self.batch_size = tf.shape(images_ph)[0]
    # self.original_size = config.original_size
    # self.num_channels = config.num_channels
    # self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    # self.minRadius = config.minRadius
    # self.depth = config.depth
    #
    # self.hg_size = config.hg_size
    # self.hl_size = config.hl_size
    # self.g_size = config.g_size
    # self.loc_dim = config.loc_dim
    #
    # self.images_ph = images_ph

    self.init_weights()

  def init_weights(self):
    self.lookup = tf.Variable(
          tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
          name="lookup")

    # """ Initialize all the trainable weights."""
    # self.w_g0 = weight_variable((self.config.sensor_size, self.config.hg_size))
    # self.b_g0 = bias_variable((self.config.hg_size,))
    # self.w_l0 = weight_variable((self.config.loc_dim, self.config.hl_size))
    # self.b_l0 = bias_variable((self.config.hl_size,))
    # self.w_g1 = weight_variable((self.config.hg_size, self.config.g_size))
    # self.b_g1 = bias_variable((self.config.g_size,))
    # self.w_l1 = weight_variable((self.config.hl_size, self.config.g_size))
    # self.b_l1 = weight_variable((self.config.g_size,))
    self.W = []
    self.b = []
    for i, filter_size in enumerate(self.config.filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W"+str(i))
        b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b"+str(i))
        self.W.append(W)
        self.b.append(b)

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    #calculate input_x from location
    with tf.name_scope("PRE_GLIMPSE"):

      # img_list = tf.split(value= self.images_ph, num_split= self.batch_size, split_dim= 0)


      # # tr_image = tf.transpose(self.images_ph)
      # initial_loc = tf.sub(tf.cast(loc, dtype= tf.int32), tf.constant(1, dtype= tf.int32))
      # x_pad = tf.pad(initial_loc, [[0, 0, ], [0, 1]])
      # #@To-Do : verify values
      # x_pad = tf.reshape(x_pad, shape= [-1, 2, 1])
      # begin_tensor = tf.cast(x_pad, dtype= tf.int32)
      #
      # s_1 = tf.zeros([self.batch_size], 3)
      # s_1 = tf.reshape(s_1, shape= [-1, 1])
      # size_pad_1  = tf.pad(s_1, [[0, 0, ], [0, 1]])
      #
      # s_2 = tf.zeros([self.batch_size], self.config.embedding_size)
      # s_2 = tf.reshape(s_2, shape=[-1, 1])
      # size_pad_2 = tf.pad(s_2, [[0, 0, ], [1, 0]])
      #
      # size_pad_ = tf.add(size_pad_1, size_pad_2)
      # size_tensor = tf.reshape(size_pad_, shape= [-1, 2, 1])
      # size_tensor = tf.cast(size_tensor, dtype= tf.int32)
      #
      # s_2 = tf.zeros(self.batch_size, self.config.embedding_size)
      # size_pad_2 = tf.add(size_pad_1, tf.zeros(self.batch_size, 3))
      # size_pad_x = tf.pad(size_pad_2, [[0, 0, ], [0, 1]])
      #
      # y_ = tf.constant(128, shape=tf.shape(size_pad_2))
      # size_pad_y = tf.pad(y_, [[0, 0, ], [1, 0]])
      #
      # size_tensor = tf.add(size_pad_x, size_pad_y)
      # # x_1 = tf.mul(size_pad, tf.constant(0, shape= tf.shape(size_pad)))
      # # add_x
      # # size_tensor = [self.batch_size, tf.constant(3), self.]
      # input_x = tf.slice(input_= self.images_ph, begin= begin_tensor, size= size_tensor)

      # return input_x
      # # Embedding layer
      # with tf.device('/cpu:0'), tf.name_scope("embedding"):
      #   W = tf.Variable(
      #     tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
      #     name="W")
      #   embedded_chars = tf.nn.embedding_lookup(W, input_x)
      #   embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
      #   return embedded_chars_expanded
      imgs = tf.reshape(self.images_ph, [
          tf.shape(self.images_ph)[0], self.config.sequence_length, self.config.embedding_size])
      imgs = tf.expand_dims(imgs, -1)
      glimpse_imgs = tf.image.extract_glimpse(imgs,
                                              [self.win_size, self.config.embedding_size], loc)
      glimpse_imgs = tf.reshape(glimpse_imgs, [
          tf.shape(loc)[0], self.win_size , self.config.embedding_size,1
      ])
      glimpse_imgs = tf.cast(glimpse_imgs, dtype= tf.float32)
      return glimpse_imgs

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(self.config.filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            # filter_shape = [filter_size, embedding_size, 1, num_filters]
            # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                glimpse_input,
                self.W[i],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.b[i]), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.config.win_size - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
    h_pool = tf.concat(3, pooled_outputs)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
    return h_drop


class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.

  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
