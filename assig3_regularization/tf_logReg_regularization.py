import numpy as np
import utils.utils as utils
import tensorflow as tf

# Buscar estas constantes en codigo, se convierten en hyperparametros del modelo
#l2_regularization_factor=0.01
#learnig_rate=0.5

"""
Usando Placeholder para alimentar con los datos de entrada. Ya no fijos, selecciono un subconjunto en cada
paso, para implementar el SGD.
"""
def multi_log_reg_SGD_regularization(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                                     image_size, num_labels, batch_size=128, num_steps=3001):
    # Build Graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        """El placeholder correspondiente a las entradas para entrenamiento se alimenta en ejecucion
        con el minibatch seleccionado en cada iteracion para el SGD
        Ya no es constante"""
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        l2_factor = 0.01
        loss_l2 = loss + l2_factor * tf.nn.l2_loss(weights)

        # Optimizer.
        # 0.5 es el learning rate
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_l2)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    # Execute Graph
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
                #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                print('Validation accuracy: %.1f%%' % utils.accuracy(session.run(valid_prediction), valid_labels))
        print("Test accuracy: %.1f%%" % utils.accuracy(test_prediction.eval(), test_labels))