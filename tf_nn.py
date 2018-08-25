import json
import tqdm
from math import sqrt
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


def dnn(x, _num_features, n_classes):
    # Define network architecture
    input_size = _num_features
    n_nodes_hl = [500] * 3  # hidden layer sizes
    activation_fn = [tf.nn.relu] * 3  # allows for different activation fn per layer
    architecture = [input_size] + n_nodes_hl + [n_classes]  # concatenate lists

    # Define layer variables
    layers = []
    for i, n_nodes in enumerate(architecture):
        if i == 0:
            continue  # can't reference previous size

        layers.append({
            'weights': tf.Variable(tf.random_normal(shape=[architecture[i - 1], n_nodes])),
            'biases': tf.Variable(tf.random_normal(shape=[n_nodes]))
        })

    # Define model for feed forward
    model = None
    for i, layer in enumerate(layers[:-1]):  # leave out output layer (only input & hidden layers)
        _input = x if i == 0 else model
        model = activation_fn[i](tf.matmul(_input, layer['weights']) + layer['biases'])

    # Return output
    output_layer = layers[-1]
    return tf.matmul(model, output_layer['weights']) + output_layer['biases']


def rnn_lstm(x, _num_features, _n_classes, **kwargs):
    # Set parameter default
    rnn_size = kwargs['rnn_size'] or 128

    # Reshape x
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, shape=[-1, kwargs['chunk_size']])
    x = tf.split(x, kwargs['n_chunks'])

    # Create LSTM cell
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Define layer variables
    weights = tf.Variable(tf.random_normal(shape=[rnn_size, _n_classes])),
    biases = tf.Variable(tf.random_normal(shape=[_n_classes]))

    # Return output
    return tf.matmul(outputs[-1:], weights) + biases


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                          size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x, _num_features, _n_classes, **kwargs):
    # Set parameter default
    keep_rate = kwargs['keep_rate'] or 0.8

    # Reshape x (from flat image to 28x28 image
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Define layer variables
    weights = {
        # 5x5 convolution, 1 input, will output 32 features
        'W_conv_1': tf.Variable(tf.random_normal(shape=[5, 5, 1, 32])),
        # 5x5 convolution, 32 inputs, will output 64 features
        'W_conv_2': tf.Variable(tf.random_normal(shape=[5, 5, 32, 64])),
        # Each convolution layer downsamples the image in half: 28x28 - 14x14 - 7x7
        # 7x7 image (after convolution), 64 features, 1024 nodes in dense layer
        'W_fc': tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 1024])),
        'out': tf.Variable(tf.random_normal(shape=[1024, _n_classes])),
    }

    biases = {
        'b_conv_1': tf.Variable(tf.random_normal(shape=[32])),
        'b_conv_2': tf.Variable(tf.random_normal(shape=[64])),
        'b_fc': tf.Variable(tf.random_normal(shape=[1024])),
        'out': tf.Variable(tf.random_normal(shape=[_n_classes])),
    }

    # Create convolution (with pool) layers
    conv_1 = tf.nn.relu(conv2d(x, weights['W_conv_1']) + biases['b_conv_1'])
    conv_1 = maxpool2d(conv_1)

    conv_2 = tf.nn.relu(conv2d(conv_1, weights['W_conv_2']) + biases['b_conv_2'])
    conv_2 = maxpool2d(conv_2)

    # Create fully connected layer
    fc = tf.reshape(conv_2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    # Apply dropout regularization on the fc layer
    fc = tf.nn.dropout(fc, keep_rate)

    # Return output
    return tf.matmul(fc, weights['out']) + biases['out']


def train_neural_network(_data, model_fn, n_epochs=10, batch_sz=100, do_chunks=False, unpack_sparse=False, **kwargs):
    num_samples, num_features = _data['train']['x'].shape
    n_classes = _data['train']['y'].shape[1]

    n_chunks, chunk_size = None, None
    if do_chunks:
        chunk_size = int(sqrt(num_features))
        n_chunks = num_features // chunk_size

        kwargs['chunk_size'] = chunk_size
        kwargs['n_chunks'] = n_chunks

        # chunk_size * n_chunks likely to be < num_features so
        # we update it & the training data (drop a few features)
        num_features = chunk_size * n_chunks
        _data['train']['x'] = _data['train']['x'][:num_features]
        _data['train']['y'] = _data['train']['y'][:num_features]

        # Create placeholder for x
        x = tf.placeholder('float', shape=[None, n_chunks, chunk_size])

    else:
        # Create placeholder for x
        x = tf.placeholder('float', shape=[None, num_features])

    # Create placeholder for y
    y = tf.placeholder('float')

    # Define tf operations
    prediction = model_fn(x, num_features, n_classes, **kwargs)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimiser = tf.train.AdamOptimizer().minimize(cost)

    # Define scoring function
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # Start tensorflow session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        losses = []

        # Train
        with tqdm.tqdm(total=n_epochs, unit='epoch') as pbar:
            for epoch in range(n_epochs):
                train_loss = 0

                # In batches
                for i in range(num_samples // batch_sz):
                    x_batch = _data['train']['x'][i * batch_sz:(i * batch_sz + batch_sz), ]
                    y_batch = _data['train']['y'][i * batch_sz:(i * batch_sz + batch_sz), ]

                    if unpack_sparse:
                        # The data is made out of sparse matrices.
                        # We need to convert to arrays before passing it to tf, hence `.toarray()`
                        x_batch = x_batch.toarray()
                        y_batch = y_batch.toarray()

                    # TODO: LSTM (RNN) model is buggy (chunking part during training)
                    if do_chunks:
                        # Reshape x batch
                        x_batch = x_batch.reshape((batch_sz, n_chunks, chunk_size))

                    _, c = session.run([optimiser, cost], feed_dict={x: x_batch, y: y_batch})
                    train_loss += c

                losses.append(train_loss)
                pbar.set_description('Epoch {0:>3} / {1} -> Loss: {2}'.format(epoch + 1, n_epochs, train_loss))
                pbar.update()

        # Limit the size of test set to prevent out of GPU VRAM errors
        _x_test = _data['test']['x'][:256]
        _y_test = _data['test']['y'][:256]

        if unpack_sparse:
            # The data is made out of sparse matrices.
            # We need to convert to arrays before passing it to tf, hence `.toarray()`
            _x_test = _x_test.toarray()
            _y_test = _y_test.toarray()

        # TODO: LSTM (RNN) model is buggy (chunking part during training)
        if do_chunks:
            # Reshape arrays
            _x_test = _x_test.reshape((-1, n_chunks, chunk_size))
            _y_test = _y_test.reshape((-1, n_chunks, chunk_size))

        acc = accuracy.eval({x: _x_test, y: _y_test})

    return acc, losses


def get_lexicon_details(f_in):
    with open(f_in, 'r') as f:
        content = f.read()
        granularity, size = content.split(',')

    return granularity, size


def load_small_data():
    # Load lexicon
    with open('preprocessing/small_data/saved/lexicon.json', 'r', encoding='latin-1') as f:
        _lexicon = json.load(f)

    # Load sparse matrices
    _x_train = load_npz('preprocessing/small_data/saved/x_train.npz')
    _y_train = load_npz('preprocessing/small_data/saved/y_train.npz')
    _x_test = load_npz('preprocessing/small_data/saved/x_test.npz')
    _y_test = load_npz('preprocessing/small_data/saved/y_test.npz')

    return _lexicon, _x_train, _y_train, _x_test, _y_test


def load_large_data(test_size=0.1):
    # Filenames & lexicon details
    lexicon_details_filename = 'preprocessing/large_data/saved/lexicon-details.csv'

    lexicon_granularity, lexicon_size = get_lexicon_details(lexicon_details_filename)

    lexicon_filename = 'preprocessing/large_data/saved/lexicon-{}-{}.json'.format(lexicon_granularity, lexicon_size)
    features_filename = 'preprocessing/large_data/saved/processed-train-set-{}-{}-features.npz' \
        .format(lexicon_granularity, lexicon_size)
    labels_filename = 'preprocessing/large_data/saved/processed-train-set-{}-{}-labels.npz' \
        .format(lexicon_granularity, lexicon_size)

    # Start loading stuff
    print('Loaded lexicon details from `{}` -> granularity: {}\tsize: {}'
          .format(lexicon_details_filename, lexicon_granularity, lexicon_size))

    # Load lexicon
    print('-- loading lexicon from `{}`'.format(lexicon_filename))
    with open(lexicon_filename, 'r', encoding='latin-1') as f:
        _lexicon = json.load(f)

    # Load sparse matrices
    print('-- loading features from `{}'.format(features_filename))
    features = load_npz(features_filename)

    print('-- loading labels from `{}'.format(labels_filename))
    labels = load_npz(labels_filename)

    # Split train & test sets
    print('-- creating train & test sets...')
    testing_size = int(test_size * features.shape[0])
    _x_train = features[:-testing_size]
    _y_train = labels[:-testing_size]
    _x_test = features[-testing_size:]
    _y_test = labels[-testing_size:]

    return _lexicon, _x_train, _y_train, _x_test, _y_test


def load_mnist_data():
    mnist = input_data.read_data_sets(r'D:\tf_data', one_hot=True)

    _x_train = mnist.train.images
    _y_train = mnist.train.labels
    _x_test = mnist.test.images
    _y_test = mnist.test.labels

    return _x_train, _y_train, _x_test, _y_test


if __name__ == '__main__':
    # Set program parameters
    SMALL_DATA = False
    USE_MNIST = True

    MODELS = [
        {'fn': 'dnn', 'kwargs': {}},
        {'fn': 'rnn_lstm', 'kwargs': {'do_chunks': True, 'rnn_size': 128}},
        {'fn': 'cnn', 'kwargs': {'keep_rate': 0.8}},
    ]

    for k, model in enumerate(MODELS):
        # MNIST data does not use sparse matrices
        model['kwargs']['unpack_sparse'] = False if USE_MNIST else True
        MODELS[k] = model

    # Load data
    print('Loading data...')
    if SMALL_DATA and not USE_MNIST:
        lexicon, x_train, y_train, x_test, y_test = load_small_data()
    elif USE_MNIST:
        x_train, y_train, x_test, y_test = load_mnist_data()
    else:
        # For computing the accuracy on the large data test set with test_size of 0.1 (10%) you will need at
        # least 2GB of GPU RAM. I'm running this on a laptop with 1GB of GPU RAM so I'm only using 1% for testing.
        lexicon, x_train, y_train, x_test, y_test = load_large_data(test_size=0.01)

    # Set training parameters
    data = {
        "train": {"x": x_train, "y": y_train},
        "test": {"x": x_test, "y": y_test},
    }

    # TODO: LSTM (RNN) model is buggy (chunking part during training)
    network_model = MODELS[2]  # (0 - DNN, 1 - LSTM (RNN))

    if USE_MNIST:
        num_epochs, batch_size = 10, 128
    else:
        # Be careful when setting the batch size.
        # A low batch size will have a small memory footprint but training will take forever.
        # A large batch size might lead to a suboptimal accuracy do to less training.
        # (remember we do backpropagation after each batch)
        #
        # For a problem (x_train) with 2461 features, a 10K batch size will take about 180MB of RAM
        # A batch size of 100K will take about 1.75GB of RAM (and 1.75GB of GPU RAM).
        #
        # This is because during training we must unpack the batch (sparse matrix) into a numpy array.
        num_epochs, batch_size = (100, 100) if SMALL_DATA else (10, 20000)

    # Train network
    print('\nTraining network `{}` ->\tx_train size: {} x {}\t\tEpochs: {}\t\tBatch size: {}\n'
          .format(network_model['fn'], x_train.shape[0], x_train.shape[1], num_epochs, batch_size))

    _accuracy, train_losses = train_neural_network(data,
                                                   eval(network_model['fn']),
                                                   num_epochs, batch_size,
                                                   **(network_model['kwargs']))

    # Print results
    print('Network accuracy:', _accuracy)

    # Plot train error (loss)
    plt.plot(np.arange(len(train_losses)) + 1, train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Train Error')
    plt.title('Train Error over Epochs')
    plt.show()
