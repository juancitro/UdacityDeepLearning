import numpy as np
import utils.utils as utils
import tensorflow as tf

def multinomial_logistic_regression_GD(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                                       image_size, num_labels, train_subset=10000, num_steps=801):
    # Subset the training data for faster turnaround.
    #train_subset = 10000
    """Build Graph"""
    graph = tf.Graph()
    """ Dentro del WITH, se inicializan todos aspectos del grafo """
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        """ Porque a cero los biases? Standford los ponia en 1 """
        ## Weights y Biases se usan para logits, loss y prediction
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        """ logits es 'y' """
        """ loss: funcion de costo a optimizar"""
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        # 0.5 es el learning rate
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            """l: loss value;  predictions: training predictions"""
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            # Calcula valores cada 100 pasos
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
                print(
                'Training accuracy: %.1f%%' % utils.accuracy(predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
                print('Validation accuracy: %.1f%%' % utils.accuracy(session.run(valid_prediction), valid_labels))
        print('Test accuracy: %.1f%%' % utils.accuracy(test_prediction.eval(), test_labels))


"""
Usando Placeholder para alimentar con los datos de entrada. Ya no fijos, selecciono un subconjunto en cada
paso, para implementar el SGD.
"""
def multinomial_logistic_regression_SGD(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
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

        # Optimizer.
        # 0.5 es el learning rate
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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