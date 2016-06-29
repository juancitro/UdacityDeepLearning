import numpy as np
import utils.utils as utils
import tensorflow as tf

def nn_h1(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                                       image_size, num_labels, batch_size=128, num_steps=3001):
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        unit_l1 = 1024

        #Model Parameters
        weights_l1 = tf.Variable(tf.truncated_normal([image_size * image_size, unit_l1]))
        biases_l1 = tf.Variable(tf.zeros([unit_l1]))

        #Hidden Layer
        z1 = tf.matmul(tf_train_dataset,weights_l1) + biases_l1
        h1 = tf.nn.relu(z1)

        #Model Parameters
        weights_l2 = tf.Variable(tf.truncated_normal([unit_l1, num_labels]))
        biases_l2 = tf.Variable(tf.zeros([num_labels]))

        #Output Layer, Computations
        logits = tf.matmul(h1, weights_l2) + biases_l2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        #Optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        h1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights_l1) + biases_l1)
        valid_prediction = tf.nn.softmax(tf.matmul(h1_valid, weights_l2) + biases_l2)
        h1_test = tf.nn.relu(tf.matmul(tf_test_dataset,weights_l1) + biases_l1)
        test_prediction = tf.nn.softmax(tf.matmul(h1_test, weights_l2) + biases_l2)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % utils.accuracy(predictions, batch_labels))
                # print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                print('Validation accuracy: %.1f%%' % utils.accuracy(session.run(valid_prediction), valid_labels))
        print("Test accuracy: %.1f%%" % utils.accuracy(test_prediction.eval(), test_labels))
