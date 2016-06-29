# coding=utf-8
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image

# modulo dedicado al procesamiento de imágenes (imagenes n-dimensionales)
from scipy import ndimage

from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve

# Python object serialization
from six.moves import cPickle as pickle

from PIL import Image

num_clases = 10
image_size = 28 # Pixel width and height.
pixel_depth = 255.0 # Number of levels per pixel.


# Devuelve una lista con las carpetas de un directorio base (carpetas con las clases)
def data_folders(root_folder):
    folders = [os.path.join(root_folder, d) for d in (sorted(os.listdir(root_folder)))]
    if len(folders) != num_clases:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (num_clases, len(folders)))
    return folders


"""
# Imagenes por letra.
# Convierte las el conjunto de imagenes a una matriz de 3D (image index, x, y) of floating point values, normalized to have
# approximately zero mean and standard deviation ~0.5
# RETORNA ARREGLO 3D (image index, x, y) CON LAS IMÁGENES POR LETRA.
# un NDARRAY
"""
def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)

    # Matriz de 3D, con cada imagen.
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            # Normalizando el valor del pixel con la formula:
            # (pixelValue-128)/128
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


"""
# Convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have
# You might not be able to fit it all in memory, we'll load each class into a separate dataset (llamando a load_letter),
# store them on disk (this method).
# and curate them independently. Later we'll merge them into a single dataset of manageable size.
# SERIALIZA LOS OBJETOS
# GUARDA EN UN ARCHIVO EL DATASET DE CADA LETRA
# RETORNA LOS NOMBRES DE LOS ARCHIVOS CREADOS
"""
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    # Write a pickled representation of obj(dataset) to the open file object file(f).
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names


def load_from_pickle_file(pickleFileName):
    dataset = None
    if os.path.exists(pickleFileName):
        try:
            with open(pickleFileName, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print('Unable to load data from', pickleFileName, ':', e)
    else:
        print('File not exist')
    #print(dataset.shape)
    return dataset


"""
Crea arreglos con datos y clase asociada de nb_rows filas
RETURN:
    dataset: arreglo de imagenes
    labels: clase asociada a cada imagen
"""
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


"""
Recorre los archivos de imagenes serializados, y crea un conjunto de entrenamiento y uno de validacion
(si asi lo indica), segun un tamaño seleccionado.
Selecciona "size/num_clases" imagenes de cada letra.
PARAMETROS:
    pickle_files: lista archivos serializados
    train_size: tamaño del conjunto de entrenamiento
    valid_size: tamaño del conjunto de validacion (a 0 porque no le interesa?)
RETURN:
    valid_dataset: arreglo de imagenes para validacion
    valid_labels: clase asociada a cada imagen del arreglo anterior
    train_dataset: arreglo de imagenes para entrenamiento
    train_labels: clase asociada a cada imagen del arreglo de entrenamiento
"""
def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    # Itera por letra
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                # ndarray de la letra, conjunto de imagenes para esa letra.
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    # Selecciona un subconjunto de imagenes de letras del tamaño elegido
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    # Las agrega al conjunto de validacion, las imagenes y la clase
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    # Actualiza los indices del conjunto de validacion para cargar nuevas imagenes de otras letras
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                #Selecciona un subconjunto de imagenes de la letra actual, del tamaño elegido
                # Distintas de la de validacion
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                # Las agrega al conjunto de entrenamiento, imagenes y clase
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                # Actualiza los indices para cargar nuevas imagenes a continuacion
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


"""
Randomizar conjunto de imagenes y sus clases asociadas
"""
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

"""
Save data
Guarda todos los datos del preprocesamiento
"""
def save_data(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    pickle_file = 'notMNIST.pickle'
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    return pickle_file