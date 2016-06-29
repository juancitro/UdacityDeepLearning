from __future__ import print_function

import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import os

import numpy as np
#from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
#import six.moves.cPickle as pickle
#from PIL import Image
import preprocess_data as pre_d

# Config the matlotlib backend as plotting inline in IPython
#matplotlib inline

train_root_folder = 'Data/notMNIST_large/'
test_root_folder = 'Data/notMNIST_small/'

#file = 'Data/notMNIST_large/A/a2F6b28udHRm.png'
def show_image(file):
    image = mpimg.imread(file)
    mpplt.imshow(image)
    mpplt.show()

"""
Genera los archivos .pickle por cada letra a partir de un directorio raiz.
Devuelve la lista de los archivos generados
"""
def generate_pickles_files(root_folder, min_num_images_per_class):
    folders = pre_d.data_folders(root_folder)
    datasets = pre_d.maybe_pickle(folders, min_num_images_per_class)
    return datasets

"""
Devuelve la lista de archivos con los datasets .pickle a partir de un directorio base
Un archivo .pickle por cada letra
"""
def get_dataset_from_pickles(root_folder):
    datasets = []
    for file in os.listdir(root_folder):
        if file.endswith(".pickle"):
            datasets.append(root_folder+file)
    return datasets

"""
Genera un modelo de regresion logistica a partir de un conjunto de entrenamiento.
PARAMETROS:
    train_datasets: archivos .pickle de las letras
    train_size: tamanio del conjunto de entrenamiento
RETURN:
    logReg: modelo de regresion logistica
"""
def regresion_logistica(train_dataset, train_labels, train_size=1000):
    # Plancha X, la imagen la convierte en un arreglo
    X = np.ndarray(shape=(train_dataset.shape[0], train_dataset.shape[1]*train_dataset.shape[2]),
                             dtype=np.float32)
    for i in range(0,train_dataset.shape[0]) :
        X[i] = train_dataset[i,:,:].flatten()

    logReg = LogisticRegression()
    logReg.fit(X,train_labels)
    return logReg


"""
Guardar datasets
"""
def generate_save_all_data():
    #train_datasets = generate_pickles_files(train_root_folder,45000)
    #test_datasets= generate_pickles_files(test_root_folder,1800)
    train_datasets = get_dataset_from_pickles(train_root_folder)
    test_datasets = get_dataset_from_pickles(test_root_folder)
    train_size = 200000
    valid_size = 10000
    test_size = 10000
    valid_dataset, valid_labels, train_dataset, train_labels = pre_d.merge_datasets(train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = pre_d.merge_datasets(test_datasets, test_size)
    # Randomize los conjuntos de imagenes
    train_dataset, train_labels = pre_d.randomize(train_dataset, train_labels)
    test_dataset, test_labels = pre_d.randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = pre_d.randomize(valid_dataset, valid_labels)
    #CHEQUEAR RUTAAAAAA
    pre_d.save_data(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)


if __name__ == "__main__":
    generate_save_all_data()
    # train_datasets = get_dataset_from_pickles(train_root_folder)
    # test_datasets = get_dataset_from_pickles(test_root_folder)
    # train_size = 100
    # valid_size = 50
    # test_size = 50
    # valid_dataset, valid_labels, train_dataset, train_labels = pre_d.merge_datasets(train_datasets, train_size,
    #                                                                                 valid_size)
    # _, _, test_dataset, test_labels = pre_d.merge_datasets(test_datasets, test_size)
    # model = regresion_logistica(train_dataset,train_labels,train_size)
    # """model.score: Para un model de LogisticRegression, un dataset y sus labels reales, devuelve
    # la precision del modelo."""
    # train_dataset_flat = train_dataset.reshape((-1,784))
    # valid_dataset_flat = valid_dataset.reshape((-1,784))
    # test_dataset_flat = test_dataset.reshape((-1,784))
    # print('Train score: ' + str(model.score(train_dataset_flat, train_labels)))
    # print('Valid score: ' + str(model.score(valid_dataset_flat, valid_labels)))
    # print('Test score: ' + str(model.score(test_dataset_flat,test_labels)))
    #
    # """Para un model LogisticRegression y un conjunto de entrada, devuelve las predicciones."""
    # #model.predict(test_dataset)
