#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : rcnn
Describe: 
    
Author : LH
Date : 2019/9/29
"""
import time
import random
import numpy as np
import tensorflow as tf

from basemodel import BaseModel


def today_str():
    current_time = time.localtime()
    today = time.strftime('%Y%m%d%H%M', current_time)
    return today


def get_values(kwargs, key, default_value):
    if kwargs.__contains__(key):
        return kwargs[key]
    return default_value


class RCNN(BaseModel):
    def __init__(self, trainable=True, sequence_length=100, label_nums=10, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self._trainable = trainable
        tf.logging.set_verbosity(tf.logging.INFO)
        # 生成默认的
        if self._trainable:
            self._sequence_length = sequence_length
            self._label_nums = label_nums

            self._display_num = get_values(kwargs, 'display_num', 10)
            self._batch_size = get_values(kwargs, 'batch_szie', 32)
            self._epochs = get_values(kwargs, 'epochs', 64)
            self._steps = get_values(kwargs, 'steps', 200)
            self._save_path = get_values(kwargs, 'save_path', './model/rcnn/')
            self._model_name = get_values(kwargs, 'model_name', 'rcnn' + today_str())
            self._vocab_size = get_values(kwargs, 'vocab_size', 300000)
            self._embedding_size = get_values(kwargs, 'embedding_size', 200)
            self._represent_size = get_values(kwargs,'represent_size',300)
            self._rnn_layers_size = get_values(kwargs, 'rnn_layers_size', [128, 128])
            self._dnn_layers_size = get_values(kwargs, 'dnn_layers_size', [1024, 256])

            with tf.name_scope('input'):
                self._input_features = tf.placeholder(dtype=tf.int32, shape=[None, self._sequence_length],
                                                      name='input_features')
                self._input_labels = tf.placeholder(dtype=tf.int32, shape=[None, self._label_nums], name='input_labels')
                self._dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

            with tf.name_scope('embedding'):
                W = tf.Variable(
                    tf.random_uniform(shape=[self._vocab_size, self._embedding_size], minval=-0.1, maxval=0.1
                                      , dtype=tf.float32), name='W')
                self._embedding_input = tf.nn.embedding_lookup(W, self._input_features, name='features')
                self._embedding_input_copy = self._embedding_input
            # 双向rnn计算
            fwList = []
            bwList = []
            # 计算合并后的长度
            pre_layer_size = self._rnn_layers_size[0] * 2
            for rnnix, rnnlayersize in enumerate(self._rnn_layers_size):
                with tf.name_scope('bi-lstm-{}'.format(rnnix)):
                    # 加RNN单元的长度
                    fwlstmcell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=rnnlayersize),
                                                               output_keep_prob=self._dropout_keep_prob)
                    bwlstmcell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=rnnlayersize),
                                                               output_keep_prob=self._dropout_keep_prob)
                    fwList.append(fwlstmcell)
                    bwList.append(bwlstmcell)
            # 多层RNN
            fwMulCells = tf.nn.rnn_cell.MultiRNNCell(cells=fwList)
            bwMulCells = tf.nn.rnn_cell.MultiRNNCell(cells=bwList)
            outputs, self._current_state = tf.nn.bidirectional_dynamic_rnn(fwMulCells, bwMulCells,
                                                                           self._embedding_input, dtype=tf.float32)
            fwOutput, bwOutput = outputs
            # 加词向量的长度
            pre_layer_size += self._embedding_size
            with tf.name_scope('representation'):
                # 合并双向lstm的输出及词向量
                represent = tf.concat([fwOutput, self._embedding_input_copy, bwOutput], axis=2)
                W = tf.Variable(tf.random_uniform(shape=[pre_layer_size, self._represent_size], dtype=tf.float32),
                                        name='W')
                b = tf.constant(0.0,shape=[self._represent_size],name='b')
                self._represent = tf.expand_dims(tf.nn.tanh(tf.einsum('aij,jk->aik',represent,W)+b,name='represent'),axis=3)
                # 最大池化1维度，得到最大池化后的表示，并最后去掉1，3维度
                self._max_pool_represent = tf.squeeze(
                    tf.nn.max_pool(self._represent, ksize=[1, self._sequence_length, 1, 1],
                                   strides=[1, 1, 1, 1], padding='VALID'), axis=[1, 3])
            pre_layer_size = self._represent_size
            dnn_output = self._max_pool_represent
            # with tf.name_scope('dnn'):
            #     for dnnix, dnnlayersize in enumerate(self._dnn_layers_size):
            #         with tf.name_scope('dnn-{}'.format(dnnix)):
            #             W = tf.Variable(tf.random_uniform(shape=[pre_layer_size, dnnlayersize], dtype=tf.float32),
            #                             name='W')
            #             b = tf.constant(0.0, shape=[dnnlayersize], name='b')
            #             h = tf.nn.tanh(tf.nn.xw_plus_b(dnn_output, W, b), name='h')
            #             dnn_output = h
            #             # 替换为这一层神经元大小，方便下一次计算
            #             pre_layer_size = dnnlayersize

            with tf.name_scope('prediction'):
                W = tf.Variable(tf.random_uniform(shape=[pre_layer_size, self._label_nums], dtype=tf.float32), name='W')
                b = tf.constant(0.0, shape=[self._label_nums], name='b')
                # 先计算未softmax的输出值
                self._prob_prediction = tf.nn.xw_plus_b(dnn_output, W, b)
                self._one_hot_prediction = tf.argmax(self._prob_prediction, 1, name='one_hot_prediction')

            with tf.name_scope('loss'):
                # 计算softmax输出后的loss
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._prob_prediction,
                                                                           labels=tf.cast(self._input_labels,
                                                                                          dtype=tf.float32))
                self._train_loss = tf.reduce_mean(cross_entropy, name='train_loss')

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self._one_hot_prediction, tf.argmax(self._input_labels, 1))
                self._train_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='train_accuracy')
            self._optimizer = None
            self._train_action = None
        # 生成默认session
        log_device_placement = get_values(kwargs, 'log_device_placement', True)
        allow_soft_placement = get_values(kwargs, 'allow_soft_placement', True)
        config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                      allow_soft_placement=allow_soft_placement)
        self._sess = tf.Session(config=config_proto)

    def fit(self, **kwargs):
        # 读入训练、验证、测试数据 路径
        train_file_name = get_values(kwargs, 'train_file_name', None)
        valid_file_name = get_values(kwargs, 'valid_file_name', None)
        test_file_name = get_values(kwargs, 'test_file_name', None)
        # 若训练和测试数据为空 则报错
        if train_file_name is None:
            raise FileNotFoundError('param: train_file_name is not found')
        if valid_file_name is None:
            raise FileNotFoundError('param: valid_file_name is not found')
        train_features, train_labels = self.read(train_file_name)
        valid_x, valid_y = self.read(valid_file_name)
        # 设定优化器及优化步骤
        self._optimizer = get_values(kwargs, 'optimizer', tf.train.AdamOptimizer(1e-3))
        self._train_action = self._optimizer.minimize(self._train_loss)
        # 模型保存
        saver = tf.train.Saver()
        # 初始化变量
        self._sess.run(tf.global_variables_initializer())
        # 开始迭代
        for epoch in range(1, self._epochs + 1, 1):
            step = 1
            while step < self._steps:
                # 随机读取batch_size的训练数据
                random_index = random.randint(0, train_features.shape[0] - self._batch_size)
                train_x = train_features[random_index:(random_index + self._batch_size)]
                train_y = train_labels[random_index:(random_index + self._batch_size)]
                _, _train_loss, _train_accuracy,prob,one_hot = self._sess.run(
                    [self._train_action, self._train_loss, self._train_accuracy,self._prob_prediction,self._one_hot_prediction],
                    feed_dict={self._input_features: train_x, self._input_labels: train_y,
                               self._dropout_keep_prob: 0.5})
                # 当满足展示步数时，输入验证集进行展示
                if step % self._display_num == 0:
                    _valid_loss, _valid_accuracy = self._sess.run([self._train_loss, self._train_accuracy],
                                                                  feed_dict={self._input_features: valid_x,
                                                                             self._input_labels: valid_y,
                                                                             self._dropout_keep_prob: 1.0})
                    valid_info = 'Epoch:{:4d} Step:{:4d} train loss:{:.2f} valid loss:{:.2f} train accuracy:{:.2f} valid accuracy:{:.2f}'.format(
                        epoch, step, _train_loss, _valid_loss, _train_accuracy, _valid_accuracy)
                    tf.logging.info(valid_info)
                step += 1
            # 每个epoch保存一个模型
            saver.save(sess=self._sess, save_path=self._save_path + self._model_name, global_step=epoch)
        # 测试集不为空时进行测试集验证
        if test_file_name is not None:
            test_x, test_y = self.read(valid_file_name)
            _test_loss, _test_accuracy = self._sess.run([self._train_loss, self._train_accuracy],
                                                        feed_dict={self._input_features: test_x,
                                                                   self._input_labels: test_y,
                                                                   self._dropout_keep_prob: 1.0})
            test_info = 'Final test loss:{:.2f} test accuracy:{:.2f}'.format(_test_loss, _test_accuracy)
            tf.logging.info(test_info)
        tf.logging.info('RCNN: {} is trained.'.format(self._model_name))

    def inference(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def read(self, filename, sep=' '):
        labels = []
        features = []
        with open(filename, 'r') as f:
            for line in f:
                label = np.zeros(shape=(self._label_nums,), dtype=np.int)
                values = [int(x) for x in line.strip().split(sep=sep)]
                label[values[0]] = 1
                feature = np.array(values[1:], dtype=np.int)
                labels.append(label)
                features.append(feature)
        return np.array(features), np.array(labels)


rcnn = RCNN(sequence_length=100, label_nums=19,vocab_size=561132,display_num=5,rnn_layers_size=[100],dnn_layers_size=[512])
rcnn.fit(train_file_name='yq_train.txt', valid_file_name='yq_valid.txt')
