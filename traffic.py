import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split


# training_file = '/root/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
# testing_file = '/root/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'
#
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
#
# X_train, y_train = train['features'], train['labels']
# X_test, y_test = test['features'], test['labels']
# X_train, X_validation, y_train, y_validation = train_test_split(
#     X_train, y_train, test_size=0.1, random_state=0
# )
#
# # TODO: Number of training examples
# n_train = len(X_train)
#
# # TODO: Number of testing examples.
# n_test = len(X_test)
#
# # TODO: What's the shape of an traffic sign image?
# image_shape = X_train[0].shape
#
# # TODO: How many unique classes/labels there are in the dataset.
# n_classes = set(y_train)
# n_classes = len(list(n_classes))
#
# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)
#
# # Visualizations will be shown in the notebook.
# index = random.randint(0, len(X_train))
# image = X_train[index].squeeze()
# print(image.shape)
# print(type(image))
# plt.figure(figsize=(1,1))
# plt.imshow(image)
# print(y_train[index])


def LeNet(x, drop):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x36.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 36), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(36))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x36. Output = 14x14x36.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x128.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 36, 128), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x128. Output = 5x5x128.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x128. Output = 3200.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 3200. Output = 800.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(3200, 800), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(800))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 800. Output = 200.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(800, 200), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(200))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, drop)

    # SOLUTION: Layer 5: Fully Connected. Input = 200. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(200, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits



EPOCHS = 100
BATCH_SIZE = 128
rate = 0.001
# dropout = 0.75

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, dropout):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
#     dropout = 0.75
#
#     print("Training...")
#     print()
#
#     for i in range(EPOCHS):
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#
#         validation_accuracy = evaluate(X_validation, y_validation, dropout)
#         print("EPOCH {} ...".format(i + 1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#
#     try:
#         saver
#     except NameError:
#         saver = tf.train.Saver()
#
#     saver.save(sess, './lenet')
#     print("Model saved")
#
#
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     dropout = 1
#
#     test_accuracy = evaluate(X_test, y_test, dropout)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import glob

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./lenet.meta')
    new_saver.restore(sess, './lenet')

    predict_img = []
    image_list = []

    for image_path in glob.glob("./traffic/*.jpg"):
        image_list.append(image_path)
        cv_image = cv2.imread(image_path)
        #res_image = cv2.resize(cv_image, (32, 32))
        #print(image_path)
        #cv2.imshow('test', res_image)
        #cv2.waitKey(0)
        res_image = np.array(cv_image)

        predict_img.append(res_image)

    dropout = 1

    predict_op = tf.nn.softmax(logits)
    result = sess.run(
        predict_op, feed_dict={x: predict_img, keep_prob: dropout}
    )

    values, indices = sess.run(tf.nn.top_k(result, k=5))
    for i in range(0, len(predict_img)):
        print()
        print(image_list[i])
        for j in range(0, values[i].size):
            print("value: {0:8.2f}% <==> index: {1}".format(values[i][j]*100, indices[i][j]))

    # image_correct = X_train[indices[0][0]].squeeze()
    # print("correct label: {}".format(y_train[indices[0][0]]))
    # cv2.imshow('test2', image_correct)
    # cv2.waitKey(0)