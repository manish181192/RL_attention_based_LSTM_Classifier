import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import tensorflow as tf

class dataReader(object):
    embedding_size = 128
    filter_sizes = [3, 4, 5]
    vocab_size = None
    num_filters = 128
    sequence_length = None
    num_classes = 8
    l2_reg_lambda = 0.0
    num_gpus = 1
    TOWER_NAME = "CNN_TOWER"
    # batch_size = 10
    num_epochs = 1000
    train_filePath = "data/traindata_10_02"
    Cv_filepath = "data/testdata_10_02"
    batches =[]
    test_batches = []
    def __init__(self):
        self.count_batch = 0
        self.test_count_batch = 0
        # self.x_train,self.y_train = self.load_data_and_labels_new(self.train_filePath)

        x_text_train, self.y_train = self.load_data_and_labels_new(self.train_filePath)
        x_text_cv, self.y_dev = self.load_data_and_labels_new(self.Cv_filepath)
        self.num_train = len(x_text_train)
        self.train_batch_size = self.num_train

        self.num_test = len(x_text_cv)
        self.test_batch_size = self.num_test

        # Build vocabulary
        self.sequence_length = max([len(x.split(" ")) for x in x_text_train])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.sequence_length)
        self.x_train = np.array(list(self.vocab_processor.fit_transform(x_text_train)))
        self.x_dev = np.array(list(self.vocab_processor.fit_transform(x_text_cv)))

        # self.sequence_length = max([len(x.split(" ")) for x in x_text_train])
        # self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.sequence_length)
        # self.x_train = np.array(list(self.vocab_processor.fit_transform(self.x_train)))
        self.vocab_size = len(self.vocab_processor.vocabulary_)


        # x_dev = np.array(list(vocab_processor.fit_transform(x_text_cv)))
        data = list(zip(self.x_train,self.y_train))
        batches = self.batch_iter(data, self.train_batch_size, self.num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            self.batches.append([x_batch,y_batch])

        dev_data = list(zip(self.x_dev,self.y_dev))
        dev_batches = self.batch_iter(dev_data, self.test_batch_size, 1)

        for batch in dev_batches:
            x_batch, y_batch = zip(*batch)
            self.test_batches.append([x_batch,y_batch])


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    def load_data_and_labels(self,positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def load_data_and_labels_new(self,filepath):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        no_freebase_relations = self.num_classes
        train_examples = list(open(filepath, "r").readlines())
        train_examples = [s.strip() for s in train_examples]


        labels = np.zeros([len(train_examples),no_freebase_relations])
        x_text = []
        for i,l in enumerate(train_examples):
            splt = l.split("\t")
            pattern = splt[0]
            relation_id = splt[1]
            x_text.append(pattern)
            label = np.zeros([no_freebase_relations])
            label[int(relation_id)] = 1
            labels[i,:] = label


        # Split by words

        return [x_text, labels]


    def loadTraindata(self, filename):
        f = file(filename, 'r')
        lines = f.readlines()
        fb_id = 0
        # self.observation_matrix = np.zeros([self.no_of_relations,self.no_of_ep])
        # self.freebase_observation_matrix= np.zeros([self.no_of_freebase,self.no_of_ep])
        pattern_id = 0
        ep_id = 0
        # self.emb_matrix = np.zeros([self.no_of_relations , word_embedding_size])
        isFB = False
        ep_rel = {}
        x_train = []
        y_train = []
        pattern_count = 0
        no_of_relations = 5
        for j, l in enumerate(lines):
            if l.startswith('REL$/') == True:
                # isFB = True
                split = l.split("\t")
                pattern = split[0]
                entity1 = split[1]
                entity2 = split[2]
                enPair = entity1 + "\t" + entity2
                ep_rel[enPair] = pattern
            else:
                pattern_count += 1

        for j, l in enumerate(lines):
            if l.startswith('REL$/') == True:
                isFB = True
            split = l.split("\t")
            pattern = split[0]
            entity1 = split[1]
            entity2 = split[2]
            enPair = entity1 + "\t" + entity2
            if isFB == False:
                x_train.append(pattern)
                y_train.append(ep_rel[enPair])


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def getBatch(self):
        x_batch, y_batch = self.batches[self.count_batch]
        # x_batch = tf.pack(x_batch)
        # y_batch = tf.pack(y_batch)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        self.count_batch += 1
        return [x_batch,y_batch]
    def getBatchTest(self):
        x_batch, y_batch = self.test_batches[0]
        # x_batch = tf.pack(x_batch)
        # y_batch = tf.pack(y_batch)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        # self.test_count_batch += 1
        return [x_batch,y_batch]