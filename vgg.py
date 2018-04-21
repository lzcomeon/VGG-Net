import tensorflow as tf
import get_data

import numpy as np
import os

log_dir = "vgg_checkpoint/"
BATCH_SIZE = 32
'''

４，batch_size不可以设置的太大。目测不能超过１００，不然会出现超出显存容量的错误
'''

with tf.Graph().as_default():
    with tf.name_scope('vgg_16'):

        with tf.name_scope('conv1'):
            with tf.name_scope('conv1_1'):
                weights1_1 = tf.Variable(tf.truncated_normal([3,3,3,64], stddev = 0.1), name='weights')
                biases1_1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='biases')
            with tf.name_scope('conv1_2'):
                weights1_2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev = 0.1), name='weights')
                biases1_2 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='biases')
        with tf.name_scope('conv2'):
            with tf.name_scope('conv2_1'):
                weights2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='weights')
                biases2_1 = tf.Variable(tf.zeros([128], dtype=tf.float32), name='biases')
            with tf.name_scope('conv2_2'):
                weights2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), name='weights')
                biases2_2 = tf.Variable(tf.zeros([128], dtype=tf.float32), name='biases')
        with tf.name_scope('conv3'):
            with tf.name_scope('conv3_1'):
                weights3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name='weights')
                biases3_1 = tf.Variable(tf.zeros([256], dtype=tf.float32), name='biases')
            with tf.name_scope('conv3_2'):
                weights3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='weights')
                biases3_2 = tf.Variable(tf.zeros([256], dtype=tf.float32), name='biases')
            with tf.name_scope('conv3_3'):
                weights3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='weights')
                biases3_3 = tf.Variable(tf.zeros([256], dtype=tf.float32), name='biases')
        with tf.name_scope('conv4'):
            with tf.name_scope('conv4_1'):
                weights4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1), name='weights')
                biases4_1 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
            with tf.name_scope('conv4_2'):
                weights4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='weights')
                biases4_2 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
            with tf.name_scope('conv4_3'):
                weights4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='weights')
                biases4_3 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
        with tf.name_scope('conv5'):
            with tf.name_scope('conv5_1'):
                weights5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='weights')
                biases5_1 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
            with tf.name_scope('conv5_2'):
                weights5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='weights')
                biases5_2 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
            with tf.name_scope('conv5_3'):
                weights5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='weights')
                biases5_3 = tf.Variable(tf.zeros([512], dtype=tf.float32), name='biases')
        with tf.name_scope('fc6'):
            weights6 = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], stddev=0.1), name='weights')
            biases6 = tf.Variable(tf.zeros([4096], dtype=tf.float32), name='biases')
        with tf.name_scope('fc7'):
            weights7 = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], stddev=0.1), name='weights')
            biases7 = tf.Variable(tf.zeros([4096], dtype=tf.float32), name='biases')
        with tf.name_scope('fc8'):
            weights8 = tf.Variable(tf.truncated_normal([1, 1, 4096, 2], stddev=0.1), name='weights')
            biases8 = tf.Variable(tf.zeros([2], dtype=tf.float32), name='biases')




    def conv(input, weights, bias, padding = 'SAME'):
         conv_out = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding=padding)
         conv_out = tf.nn.bias_add(conv_out, bias)
         activation = tf.nn.relu(conv_out)

         return activation


    def vggnet(image, keep_prob):
        # weights,bias = parameters()

        conv1_1 = conv(image, weights1_1, biases1_1)
        conv1_2 = conv(conv1_1, weights1_2, biases1_2)
        max_pool1 = tf.nn.max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        conv2_1 = conv(max_pool1, weights2_1, biases2_1)
        conv2_2 = conv(conv2_1, weights2_2, biases2_2)
        max_pool2 = tf.nn.max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        conv3_1 = conv(max_pool2, weights3_1, biases3_1)
        conv3_2 = conv(conv3_1, weights3_2, biases3_2)
        conv3_3 = conv(conv3_2, weights3_3, biases3_3)
        max_pool3 = tf.nn.max_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        conv4_1 = conv(max_pool3, weights4_1, biases4_1)
        conv4_2 = conv(conv4_1, weights4_2, biases4_2)
        conv4_3 = conv(conv4_2, weights4_3, biases4_3)
        max_pool4 = tf.nn.max_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        conv5_1 = conv(max_pool4, weights5_1, biases5_1)
        conv5_2 = conv(conv5_1, weights5_2, biases5_2)
        conv5_3 = conv(conv5_2, weights5_3, biases5_3)
        max_pool5 = tf.nn.max_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

        shape_pool5 = max_pool5.get_shape()
        flatten = shape_pool5[1].value * shape_pool5[2].value * shape_pool5[3].value
        flatten = tf.reshape(max_pool5, [-1, flatten])
        w6 = tf.reshape(weights6, [7*7*512, 4096])
        # print(flatten.shape)
        fc6 = tf.nn.relu_layer(flatten, w6, biases6)
        fc6_drop = tf.nn.dropout(fc6, keep_prob)
        w7 = tf.reshape(weights7, [4096, 4096])
        fc7 = tf.nn.relu_layer(fc6_drop, w7, biases7)
        fc7_drop = tf.nn.dropout(fc7, keep_prob)
        w8 = tf.reshape(weights8, [4096, 2])

        pred_value = tf.add(tf.matmul(fc7_drop, w8), biases8)
        pred_value = tf.nn.softmax(pred_value)
        return pred_value

    # create queues for training and testing
    img_batch, label_batch = get_data.inputs('train', BATCH_SIZE)
    label_batch = tf.one_hot(label_batch, depth=2, axis=1)
    # print('-----------------1-------------------')
    # print(img_batch.shape,label_batch.shape)

    keep_prob = 0.8
    # graph for train
    logits = vggnet(img_batch, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=logits))
    optm = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    corr = tf.equal(tf.argmax(logits, 1), tf.argmax(label_batch, 1))
    acc = tf.reduce_mean(tf.cast(corr, tf.float32))

    #
    # vs = tf.trainable_variables()
    # for v in vs:
    #     print(v)
    # exit()


    img_test, label_test1 = get_data.inputs('test', BATCH_SIZE)
    label_test = tf.one_hot(label_test1, depth=2, axis=1)

    # graph for test
    logits_test = vggnet(img_test, 1)
    cost_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=logits_test))
    corr_test = tf.equal(tf.argmax(logits_test, 1), tf.argmax(label_test, 1))
    acc_test = tf.reduce_mean(tf.cast(corr_test, tf.float32))

    # set GPU memory using model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # ckpt = tf.train.get_checkpoint_state(log_dir)
        ckpt = tf.train.get_checkpoint_state('check_point/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        # fc8 =sess.run(bias['bias_fc8'])
        #
        # print(fc8)
        # exit()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_iter = 0
        max_batch = 23000 // BATCH_SIZE
        ep = 0
        accuracy = 0.0
        loss_all = 0.0

        while not coord.should_stop():

            if num_iter <= max_batch:
                num_iter += 1
                asa,loss_in_train, acc_in_train = sess.run([optm, cost, acc])
                accuracy += acc_in_train
                loss_all += loss_in_train
                if num_iter % 15 == 0:
                    # 打印迭代１０次的结果
                    print('Epoch %d: Batch %d, loss %.3f, accuracy: %.3f' % (ep + 1, num_iter, loss_in_train, acc_in_train))
            else:
                # 打印1次Epoch的结果
                print("Epoch %d:Acc is %.3f; Loss is %.3f" % (ep + 1, accuracy / max_batch, loss_all / max_batch))
                cost_in_test, acc_in_test = sess.run([cost_test, acc_test])
                print(cost_in_test, acc_in_test)
                # 存储训练模型
                if ep % 5 == 0:
                    saver.save(sess, log_dir+'model.ckpt', global_step=ep)
                ep += 1
                num_iter = 0
                accuracy = 0
                loss = 0










