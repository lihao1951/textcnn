#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : classification
Describe: 
    新闻分类-基于tensorflow-textcnn
Author : LH
Date : 2019/9/10
"""
import tensorflow as tf
import numpy as np
import utils
import random


def xaiver_init(fin, fout, constant=1.0):
    low = -constant * tf.sqrt(6 / (fin + fout))
    high = constant * tf.sqrt(6 / (fin + fout))
    return tf.random_uniform([fin, fout], minval=low, maxval=high, dtype=tf.float32)


class TextCNN(object):
    """
    TextCNN model
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, pool_k,
                 trainable):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        if trainable:
            self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
            self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            with tf.name_scope('embedding'):
                self.W = tf.Variable(xaiver_init(vocab_size, embedding_size), name='W')
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name='b')

                    conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID",
                                        name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(h, ksize=[1, (sequence_length - filter_size + 1) / pool_k, 1, 1],
                                            strides=[1, pool_k, 1, 1], padding='VALID', name='pool')
                    pshape = pooled.get_shape().as_list()
                    repooled = tf.reshape(pooled, [-1, pshape[1] * pshape[2] * pshape[3]])
                    pooled_outputs.append(repooled)

            # num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 1)
            fshape = self.h_pool.get_shape().as_list()
            num_filters_total = fshape[1]
            with tf.name_scope('droupout'):
                self.h_drop = tf.nn.dropout(self.h_pool, keep_prob=self.dropout_keep_prob)

            with tf.name_scope('output'):
                W = tf.Variable(xaiver_init(num_filters_total, num_classes), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))
                # 增加l2损失惩罚项

                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,
                                                                    labels=tf.cast(self.input_y, dtype=tf.float32))
                self.loss = tf.reduce_mean(losses)

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            tf.logging.info("\t训练模型建立")
        else:
            self.sess = tf.Session()
            tf.logging.info("\t预测模型建立")

    def fit(self, datafiles, epochs, dropout, save_path, displayNum=20, everyEpochNum=200, batch_size=128):
        """
        对数据进行训练
        :param datafiles:
        :param batch_size:
        :param epochs:
        :param dropout:
        :param save_path:
        :return:
        """
        if len(datafiles) != 3:
            raise OSError('files`s number is wrong , must be 3')
        train_file_name = datafiles[0]
        valid_file_name = datafiles[1]
        test_file_name = datafiles[2]
        with self.sess.as_default():
            saver = tf.train.Saver(max_to_keep=3)
            train_features, train_labels = self.np_read_data(train_file_name)
            valid_features, valid_labels = self.np_read_data(valid_file_name)
            test_features, test_labels = self.np_read_data(test_file_name)
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(1, epochs + 1, 1):
                current_step = 1
                while current_step < everyEpochNum:
                    start_index = random.randint(0, train_features.shape[0] - batch_size)
                    train_x = train_features[start_index:(start_index + batch_size)]
                    train_y = train_labels[start_index:(start_index + batch_size)]
                    _, train_loss, train_accuracy = self.sess.run([self.train_op, self.loss, self.accuracy],
                                                                  feed_dict={self.input_x: train_x,
                                                                             self.input_y: train_y,
                                                                             self.dropout_keep_prob: dropout})
                    if current_step % displayNum == 0:
                        valid_loss, valid_accuracy = self.sess.run([self.loss, self.accuracy],
                                                                   feed_dict={self.input_x: valid_features,
                                                                              self.input_y: valid_labels,
                                                                              self.dropout_keep_prob: 1.0})
                        tf.logging.info(
                            ' Epoch:{:3d} Step:{:4d} train loss:{:.3f} train accuracy:{:.3f} valid loss:{:.3f} valid accuracy:{:.3f}'.
                            format(epoch, current_step, train_loss, train_accuracy, valid_loss, valid_accuracy))
                    current_step += 1

                saver.save(self.sess, save_path=save_path, global_step=epoch)
                tf.logging.info('保存模型：TextCNN-{}'.format(epoch))
            tf.logging.info("-*-训练阶段完成-*-")
            test_loss, test_accuracy = self.sess.run([self.loss, self.accuracy], feed_dict={self.input_x: test_features,
                                                                                            self.input_y: test_labels,
                                                                                            self.dropout_keep_prob: 1.0})
            tf.logging.info('Final test loss:{:.3f} test accuracy:{:.3f}'.format(test_loss, test_accuracy))

    def load_model(self, model_path):
        """
        载入模型
        :param model_path:
        :return:
        """
        path = tf.train.latest_checkpoint(model_path)
        meta_path = path + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, path)
        graph = self.sess.graph
        self.input_x = graph.get_operation_by_name('input_x').outputs[0]
        self.input_y = graph.get_operation_by_name('input_y').outputs[0]
        self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        self.predictions = graph.get_operation_by_name('output/predictions').outputs[0]
        tf.logging.info("-*-导入模型完成-*-")

    def inference(self, contlist):
        """
        预测文本分类
        :param contlist:
        :return:
        """
        test_x = contlist
        pre = self.sess.run(self.predictions, feed_dict={self.input_x: test_x, self.dropout_keep_prob: 1.0})
        return pre.tolist()

    def split_data_label(self, tline, wordnum=100, classnum=19):
        line = tf.decode_csv(tline, record_defaults=[[0] for _ in range(wordnum + 1)], field_delim=' ')
        tflabels = tf.one_hot(line[0], depth=classnum, dtype=tf.int32)
        tffeatures = tf.reshape(line[1:], shape=(wordnum,))
        return tflabels, tffeatures

    def read_data(self, filename, batch_size=64):
        textDataset = tf.data.TextLineDataset(filename)
        textDataset = textDataset.map(self.split_data_label)
        textDataset = textDataset.shuffle(buffer_size=10000)
        if batch_size > 0:
            textDataset = textDataset.batch(batch_size)
        return textDataset

    def np_read_data(self, filename, num_classes=19):
        X = []
        y = []
        with open(filename, 'r') as f:
            for line in f:
                data = line.strip().split()
                X.append(data[1:])
                l = [0 for _ in range(num_classes)]
                l[int(data[0])] = 1
                y.append(l)
        X = np.array(X, dtype=np.int)
        y = np.array(y, dtype=np.int)
        # 合并
        m = np.concatenate((X, y), axis=1)
        # 做shuffle
        np.random.shuffle(m)
        X = m[:, :-num_classes]
        y = m[:, -num_classes:]
        return X, y


def train():
    vocab = utils.vocab()
    textcnn = TextCNN(sequence_length=100, num_classes=19, vocab_size=len(vocab), embedding_size=100,
                      filter_sizes=[3, 5, 7, 9], num_filters=64, pool_k=1, trainable=True)
    textcnn.fit(['yq_train.txt', 'yq_valid.txt', 'yq_test.txt'], 30, dropout=0.5, save_path='./model/yq/lncnn',
                everyEpochNum=500, batch_size=32)

