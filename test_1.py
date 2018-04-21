import os
import tensorflow as tf
from PIL import Image


cwd = os.getcwd()  #返回当前进程的工作目录。
classes = ["cat", "dog"]

def create_record():
    writer_train = tf.python_io.TFRecordWriter("train_227.tfrecords")
    writer_test = tf.python_io.TFRecordWriter("test_227.tfrecords")
    class_path = cwd + "/train/"
    i = 0
    img_names = os.listdir(class_path)
    print(len(img_names))
    for i in range(20):
        print(img_names[i])
    for img_name in img_names:
        i += 1
        animal = img_name.split(".")[0]
        if animal == "cat":
            index = 0
        else:
            index = 1
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((227,227))
        img_raw = img.tobytes()
        #print(index,img_name)
        if i<=23000:
            example_train = tf.train.Example(
                features=tf.train.Features(feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                          "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))})
            )
            writer_train.write(example_train.SerializeToString())
        else:

            example_test = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })
            )
            #print("start to write test dataset....")
            writer_test.write(example_test.SerializeToString())

    writer_train.close()
    writer_test.close()
    exit()


data = create_record()

