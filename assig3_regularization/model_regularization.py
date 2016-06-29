import assig2_simpleModel.load_init_data as ld
import tf_logReg_regularization as tf_lrr
import tf_nn_regularization as tf_nnr
import utils.utils as utils
import numpy as np

image_size = 28
num_labels = 10
train_subset = 10000
num_steps = 801
rootFilePathData='../Data/'

if __name__ == "__main__":
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = ld.load_all_data(rootFilePathData)
    #utils.histogram('Train', train_labels)
    #utils.histogram('Valid', valid_labels)
    #utils.histogram('Test', test_labels)
    train_dataset, train_labels = ld.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = ld.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = ld.reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    #tf_lrr.multi_log_reg_SGD_regularization(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image_size, num_labels, batch_size=128, num_steps=3001)
    tf_nnr.nn_h1_regularization(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image_size,
                num_labels, batch_size=128, num_steps=3001)