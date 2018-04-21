import os
import tensorflow as tf
import numpy as np
from PIL import  Image


# def read_and_decode(filename):
#
#
#     # 创建文件队列，不限读取的数量
#     filename_queue = tf.train.string_input_producer([filename])
#     print("completed create file queue")
#     # create a reader from file queue
#     reader = tf.TFRecordReader()
#     # reader从文件队列中读入一个序列化的样本
#     _, serialized_example = reader.read(filename_queue)
#     print("completed read example")
#     #get feature from serialized example
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             "label": tf.FixedLenFeature([], tf.int64),
#             "img_raw": tf.FixedLenFeature([], tf.string)
#         }
#     )
#     print("get featured")
#     label = features["label"]
#     img = features["img_raw"]
#     img = tf.decode_raw(img, tf.uint8)
#     img = tf.reshape(img, [227, 227, 3])
#     img = tf.cast(img, tf.float32)*(1./255)*0.5
#     label = tf.cast(label, tf.int32)
#     print("over")
#
#
#     return img,label
#
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
# threads = tf.train.start_queue_runners(sess=sess)
# #image_data, label_data = read_and_decode("train.tfrecords")
# #img_batch, label_batch = tf.train.shuffle_batch([image_data,label_data], batch_size=64,capacity=100, min_after_dequeue=50)
# image_data, label_data = read_and_decode("test.tfrecords")
#
# # for i in range(10):
# #     print(image_data.shape,label_data)
# #     val,l = sess.run([image_data,label_data])
# #     print(val.shape, l)
#
# val,l = sess.run([image_data,label_data])
# print(val.shape, l)


def read_and_decode(filename, batch_size, capacity, min_after_dequeue):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "img_raw": tf.FixedLenFeature([], tf.string)
        }
    )
    label = features["label"]
    img = features["img_raw"]
    img = tf.decode_raw(img, tf.uint8)
    print("-------------1-------------")
    print(img.shape)

    img = tf.reshape(img, [227, 227, 3])
    print("-------------2-------------")
    print(img.shape)
    img = tf.cast(img, tf.float32)*1.0/255*0.5
    label = tf.cast(label, tf.int32)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)

    return img_batch, label_batch



# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# img_batch, label_batch = read_and_decode("test.tfrecords", 2000, 2, 1)
# threads = tf.train.start_queue_runners(sess=sess)
# print("-------------1-------------")
# # y = tf.keras.utils.to_categorical(label_batch, 2)
# x_image, y_label = sess.run((img_batch,label_batch))
# print(y_label.shape)
# y_label = tf.keras.utils.to_categorical(y_label, 2)
# print(y_label)
# #y = sess.run(y_label1)
# print("-------------2-------------")
#
# print(x_image.shape)
# print(y_label.shape)