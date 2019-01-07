import time
import logging
import numpy as np
import tensorflow as tf
import os


from crnn import CRNN
# from dataload import Dataload
from utlis.net_cfg_parser import parser_cfg_file
from dict_parse import Dict_Parse
from data_provider import DataProvider

os.environ["CUDA_VISIBLE_DEVICES"] = '7' #use GPU with ID=0
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(log_device_placement=False,
        allow_soft_placement=True,gpu_options=gpu_options)
class Train_CRNN(object):

    def __init__(self, pre_train=False):
        net_params, train_params = parser_cfg_file('./net.cfg')

        self.input_height = int(net_params['input_height'])
        self.input_width = int(net_params['input_width'])
        self.batch_size = int(train_params['batch_size'])
        self._learning_rate = float(train_params['learning_rate'])
        self._max_iterators = int(train_params['max_iterators'])
        self._train_logger_init()
        self._pre_train = pre_train
        self._model_save_path = str(train_params['model_save_path'])

        if self._pre_train:
            ckpt = tf.train.checkpoint_exists(self._model_save_path)
            if ckpt:
                print('Checkpoint is valid...')
                f = open('./model/train_step.txt', 'r')
                step = f.readline()
                self._start_step = int(step)
                f.close()
            else:
                assert 0
                print('Checkpoint is invalid...')
        else:
            self._start_step = 0

        data = DataProvider(32,"train-tfrecords*", 24, (60, 200), 1)
        image_batch, label_batch, class_label_batch, class_unpadded_label_batch = data.provider()
        # dict_parser = Dict_Parse("charset_size=134.txt")

        # batch_label_4 = tf.transpose(self._net_output,(1,0,2))[0:4]
        # print(type(batch_label_4),batch_label_4.shape)
        # batch_label_4 = tf.argmax(batch_label_4,2)
        # print(type(batch_label_4), batch_label_4.shape)
        # tf.summary.text('train/class_out',dict_parser.parse_array2str(batch_label_4))
        tf.summary.image('train/image', image_batch, max_outputs=4)
        tf.summary.text('train/gt_text', label_batch[:4,:])
        # tf.summary.text('train/class_label',dict_parser.parse_array2str(class_label_batch[:4,:]))


        # self._inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 1])
        self._inputs = image_batch

        # label
        # self._label = tf.sparse_placeholder(tf.int32, name='label')
        # self._label = tf.sparse_placeholder(tf.int32, name='label')
        self._label = class_unpadded_label_batch

        self._seq_len = [97] * self.batch_size

        crnn_net = CRNN(net_params, self._inputs, self._seq_len, self.batch_size, True)
        self._net_output, self._decoded, self._max_char_count = crnn_net.construct_graph()
        # net_output(97,32,134); decoded:sparse tensor; max_char_count:97

        # The length of the sequence [32] * 64
        # self._seq_len = tf.placeholder(tf.int32, [None], name='seq_len')


        self.dense_decoded = tf.sparse_tensor_to_dense(self._decoded[0], default_value=-1)

    def train(self):


        with tf.name_scope('loss'):
            # todo: 这里需要搞清楚self._label和slef._net_output的格式是否一致
            # batch_label = self._sparse_tuple_from(batch_label)   batch_label是一个列表，其中每一个元素为字符串对应的编号列表
            loss = tf.nn.ctc_loss(self._label, self._net_output, self._seq_len)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss", loss)

        with tf.name_scope('optimizer'):
            train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            accuracy = 1 - tf.reduce_mean(tf.edit_distance(tf.cast(self._decoded[0], tf.int32), self._label))
            accuracy_broad = tf.summary.scalar("accuracy", accuracy)




        # 保存模型
        saver = tf.train.Saver()

        # tensorboard
        merged = tf.summary.merge_all()

        with tf.Session(config = session_config) as sess:

            if self._pre_train:
                saver.restore(sess, self._model_save_path)
                print('load model from:', self._model_save_path)
            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter("./tensorboard_logs/", sess.graph,flush_secs = 10)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(self._start_step + 1, self._max_iterators):

                print(step)

                # feed_dict = {self._inputs: image_batch,
                #              self._label: class_unpadded_label_batch,
                #              self._seq_len: [self._max_char_count] * self.batch_size}

                # summ = sess.run(merged, feed_dict=feed_dict)
                summ = sess.run(merged)
                train_writer.add_summary(summ, global_step=step)

                # sess.run(train_op, feed_dict=feed_dict)
                sess.run(train_op)

                # print('label:', sess.run(class_unpadded_label_batch))
                # print('predict:', sess.run(self.dense_decoded, feed_dict=feed_dict))

                # print('label:', self._label.dense_shape)
                # print('net_out:',tf.shape(self._net_output))
                # print('decoded:',self._decoded[0].dense_shape)
                # print('predict:', tf.shape(self.dense_decoded))


                if step%2 == 0:
                    # train_loss = sess.run(loss, feed_dict=feed_dict)
                    train_loss = sess.run(loss)
                    self.train_logger.info('step:%d, total loss: %6f' % (step, train_loss))
                    self.train_logger.info('compute accuracy...')
                    # train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    train_accuracy = sess.run(accuracy)
                    # val_data, val_label = data.get_val_batch(self.batch_size)
                    # val_accuracy = sess.run(accuracy, feed_dict={self._inputs: val_data,
                    #                                                self._label: val_label,
                                                                   # self._seq_len: [self._max_char_count] * self.batch_size})

                    self.train_logger.info('step:%d, train accuracy: %6f' % (step, train_accuracy))
                    # self.train_logger.info('epoch:%d, val accuracy: %6f' % (epoch, val_accuracy))
                    # 用于验证网络的输出是否正确
                    # if train_accuracy>0.9:
                    #     print('label:', batch_label)
                    #     print('predict:', sess.run(self.dense_decoded, feed_dict=feed_dict))

                # if step%10 == 0:
                #     train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                #     self.train_logger.info('step:%d, train accuracy: %6f' % (epoch, train_accuracy))

                if step%100 == 0:

                    self.train_logger.info('saving model...')
                    f = open('./model/train_step.txt', 'w')
                    f.write(str(self._start_step + step))
                    f.close()
                    save_path = saver.save(sess, self._model_save_path)
                    self.train_logger.info('model saved at %s' % save_path)

                # if step%1000 == 0:
                    self.train_logger.info('compute accuracy...')
                    # train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    train_accuracy = sess.run(accuracy)
                    self.train_logger.info('step:%d, accuracy: %6f' % (step, train_accuracy))
                    # summ = sess.run(accuracy_broad, feed_dict=feed_dict)
                    summ = sess.run(accuracy_broad)
                    train_writer.add_summary(summ, global_step=step)
            coord.request_stop()
            coord.join(threads)
            train_writer.close()

    def _train_logger_init(self):
        """
        初始化log日志
        :return:
        """
        self.train_logger = logging.getLogger('train')
        self.train_logger.setLevel(logging.DEBUG)

        # 添加文件输出
        log_file = './train_logs/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.train_logger.addHandler(file_handler)

        # 添加控制台输出
        consol_handler = logging.StreamHandler()
        consol_handler.setLevel(logging.DEBUG)
        consol_formatter = logging.Formatter('%(message)s')
        consol_handler.setFormatter(consol_formatter)
        self.train_logger.addHandler(consol_handler)

    def _sparse_tuple_from(self, sequences, dtype=np.int32):
        """
        将矩阵转为稀疏矩阵存储方式
        :param sequences:
        :param dtype:
        :return:
        """

        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape
if __name__ == "__main__":
    train = Train_CRNN(pre_train=False)
    train.train()