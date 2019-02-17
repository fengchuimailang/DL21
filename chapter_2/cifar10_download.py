import cifar10
import tensorflow as tf

# tf.app.flags.FLAGS是TensorFlow内部的一个全局变量存储器，同时可以用于命令行参数的处理
FLAGS = tf.app.flags.FLAGS
FLAGS.data_dir = "cifar10_data/"

cifar10.maybe_download_and_extract()