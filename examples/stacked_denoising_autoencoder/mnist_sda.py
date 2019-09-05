"""
Example testing SDA model on MNIST digits.
"""

from sdautoencoder import SDAutoencoder
from scautoencoder import SCDAutoencoder
from softmax import test_model
from config import *
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


def get_mnist_batch_generator(is_train, batch_size, batch_limit=100):
    if is_train:
        for _ in range(batch_limit):
            yield mnist.train.next_batch(batch_size)
    else:
        for _ in range(batch_limit):
            yield mnist.test.next_batch(batch_size)


def get_mnist_batch_xs_generator(is_train, batch_size, batch_limit=100):
    for x, _ in get_mnist_batch_generator(is_train, batch_size, batch_limit):
        yield x


def sda():
    sess = tf.Session()
    sda = SDAutoencoder(dims=[784, 400, 200, 80],
                    activations=["sigmoid", "sigmoid", "sigmoid"],
                    sess=sess,
                    noise=0.20,
                    loss="cross-entropy",
                    pretrain_lr=0.0001,
                    finetune_lr=0.0001)

    mnist_train_gen_f = lambda: get_mnist_batch_xs_generator(True, batch_size=100, batch_limit=12000)

    # pretrain locally, layer by layer
    sda.pretrain_network_gen(mnist_train_gen_f)

    # fine-tune the model by training all the weights
    trained_parameters = sda.finetune_parameters_gen(get_mnist_batch_generator(True, batch_size=100, batch_limit=18000),
                                                     output_dim=10)
    mainDir = data_storage_path
    if not os.path.exists(mainDir):
        os.makedirs(mainDir)
    transformed_filepath = mainDir + "/mnist_test_transformed.csv"
    test_ys_filepath = mainDir + "/mnist_test_ys.csv"
    output_filepath = mainDir + "/mnist_pred_ys.csv"

    # for testing. write the encoded x value along with y values  to csv
    sda.write_encoded_input_with_ys(transformed_filepath, test_ys_filepath,
                                    get_mnist_batch_generator(False, batch_size=100, batch_limit=100))
    sess.close()

    test_model(parameters_dict=trained_parameters,
               input_dim=sda.output_dim,
               output_dim=10,
               x_test_filepath=transformed_filepath,
               y_test_filepath=test_ys_filepath,
               output_filepath=output_filepath)

def sca():
    sess = tf.Session()
    sca = SCDAutoencoder(dims=[784, 400, 200, 80],
                    activations=["relu", "relu", "relu"],
                    sess=sess,
                    noise=0.20,
                    loss="cross-entropy",
                    pretrain_lr=0.0001,
                    finetune_lr=0.0001)

    mnist_train_gen_f = lambda: get_mnist_batch_xs_generator(True, batch_size=100, batch_limit=12000)

    # pretrain locally, layer by layer
    sca.pretrain_network_gen(mnist_train_gen_f)

    # fine-tune the model by training all the weights
    trained_parameters = sca.finetune_parameters_gen(get_mnist_batch_generator(True, batch_size=100, batch_limit=18000),
                                                     output_dim=10)
    mainDir = data_storage_path
    if not os.path.exists(mainDir):
        os.makedirs(mainDir)
    transformed_filepath = mainDir + "/mnist_test_transformed.csv"
    test_ys_filepath = mainDir + "/mnist_test_ys.csv"
    output_filepath = mainDir + "/mnist_pred_ys.csv"

    # for testing. write the encoded x value along with y values  to csv
    sda.write_encoded_input_with_ys(transformed_filepath, test_ys_filepath,
                                    get_mnist_batch_generator(False, batch_size=100, batch_limit=100))
    sess.close()

    test_model(parameters_dict=trained_parameters,
               input_dim=sda.output_dim,
               output_dim=10,
               x_test_filepath=transformed_filepath,
               y_test_filepath=test_ys_filepath,
               output_filepath=output_filepath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sda', help='sda, sca')
    args = parser.parse_args()

    if args.mode == 'sda':
        sda()
    elif args.mode == 'sca':
        sca()
    else:
        raise Exception("Unknow --mode")
