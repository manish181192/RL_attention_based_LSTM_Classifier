from DataExtractor import dataReader

data_reader = dataReader()
class Config(object):
  win_size = 1
  depth = 1
  sensor_size = win_size**2 * depth

  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 50
  loc_std = 0.22
  original_size = data_reader.embedding_size
  num_channels = 1
  minRadius = 8
  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256
  cell_out_size = cell_size
  num_glimpses = data_reader.sequence_length
  num_classes = data_reader.num_classes
  max_grad_norm = 5.

  step = 100000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10


  #CNN
  num_epochs = data_reader.num_epochs
  filter_sizes = [1]
  embedding_size = 128
  num_filters = 128
  sequence_length = data_reader.sequence_length
  vocab_size = data_reader.vocab_size
