import os
if not os.path.exists('read'):
    os.makedirs('read/')

import tensorflow as tf

with tf.Session() as sess:
    filename = ['A.jpg','B.jpg','C.jpg']
    filename_queue = tf.train.string_input_producer(filename, shuffle=False,num_epochs = 5)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 如何取key
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i,"wb") as f:
            f.write(image_data)
