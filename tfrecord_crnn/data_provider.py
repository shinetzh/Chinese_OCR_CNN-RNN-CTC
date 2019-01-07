import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(log_device_placement=False,
        allow_soft_placement=True,gpu_options=gpu_options)

class DataProvider(object):

    def __init__(self, batch_size, tfrecord_pattern,max_sequence_length,shape=None,channels=3):
        self.batch_size = batch_size
        if shape is not None:
            self._shape = (shape[0],shape[1],channels)
        self.tfrecord_pattern = tfrecord_pattern
        self.reader = tf.TFRecordReader()
        self._max_sequence_length = max_sequence_length
        self._channels = channels
        # self._sess = tf.Session()



    def provider(self):
        files = self.get_data_files(self.tfrecord_pattern)
        filename_queue = tf.train.string_input_producer(files,shuffle = True)
        _,serialized_example = self.reader.read(filename_queue)

        feature_map = {
            'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/height':
            tf.FixedLenFeature([1], tf.int64, default_value=0),
            'image/width':
            tf.FixedLenFeature([1], tf.int64, default_value=0),
            'image/orig_width':
            tf.FixedLenFeature([1], tf.int64, default_value=0),
            'image/class':
            tf.FixedLenFeature([self._max_sequence_length], tf.int64),
            'image/unpadded_class':
            tf.VarLenFeature(tf.int64),
            'image/text':
            tf.FixedLenFeature([1], tf.string, default_value=''),
        }

        features = tf.parse_single_example(serialized_example,feature_map)
        width,height = features['image/width'], features['image/height']
        class_label = features['image/class']
        class_unpadded_label = features['image/unpadded_class']
        print(width,height)
        image = tf.image.decode_jpeg(features['image/encoded'],channels = self._channels)
        image.set_shape([None,None,self._channels])
        if self._shape is not None:
            image = tf.reshape(image,self._shape)

        label = features["image/text"]


        min_after_dequeue = 1000
        batch_size = self.batch_size
        capacity = min_after_dequeue + 3*batch_size


        image_batch, label_batch, class_label_batch,class_unpadded_label_batch = tf.train.shuffle_batch(
            [image,label,class_label,class_unpadded_label], batch_size = batch_size,
            min_after_dequeue=min_after_dequeue,capacity = capacity)
        image_batch = tf.cast(image_batch, tf.float32)
        class_unpadded_label_batch = tf.cast(class_unpadded_label_batch,tf.int32)
        return image_batch,label_batch, class_label_batch, class_unpadded_label_batch
    
    def get_data_files(self, data_sources):
        """Get data_files from data_sources.

        Args:
          data_sources: a list/tuple of files or the location of the data, i.e.
            /path/to/train@128, /path/to/train* or /tmp/.../train*

        Returns:
          a list of data_files.

        Raises:
          ValueError: if not data files are not found

        """
        if isinstance(data_sources, (list, tuple)):
            data_files = []
            for source in data_sources:
                data_files += get_data_files(source)
        else:
            if '*' in data_sources or '?' in data_sources or '[' in data_sources:
                data_files = tf.gfile.Glob(data_sources)
            else:
                data_files = [data_sources]
        if not data_files:
            raise ValueError('No data files found in %s' % (data_sources,))
        return data_files



if __name__ == "__main__":

    data = DataProvider(2, "train-tfrecords*", 24, (60, 200), 1)
    image_batch, label_batch,class_label_batch,class_unpadded_label_batch = data.provider()

    print(type(image_batch), image_batch.shape, type(label_batch), label_batch.shape)
    print(type(class_label_batch),class_label_batch.shape)
    print(type(class_unpadded_label_batch),class_unpadded_label_batch)
    Image = image_batch[0]
    label = label_batch[0]
    class_label = class_label_batch[0]
    class_unpadded_label = class_unpadded_label_batch

    print(type(Image))
    with tf.Session(config = session_config) as sess:
        # sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(4):
            print(i)
            """
            the name of [example_image,example_label] shouldn't be the same as [Image,label]

            """

            example_image, example_label,example_class_label,example_class_unpadded_label = sess.run([Image,label,class_label,class_unpadded_label])
            print(example_label)
            print(example_class_label)
            print(example_image.shape)
            print(example_class_unpadded_label)
            print(type(example_class_unpadded_label.values[0]))
            example_image = example_image.reshape(60,200)
            print(example_image.shape)
            plt.figure(1)
            plt.imshow(example_image)
            plt.show()
        coord.request_stop()
        coord.join(threads)

