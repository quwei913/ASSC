from __future__ import print_function, absolute_import, division
import glob
from collections import Counter
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed

set_random_seed(1)
from datetime import datetime
import argparse
import os
from keras.utils import to_categorical, plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adamax as opt
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import pandas as pd

from modules import *
from utils import *
from AudioDataGenerator import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--fold_idx", type=int,
                        help="data csvfile to use")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    parser.add_argument("--loadmodel",
                        help="load previous model checkpoint for retraining (Enter absolute path)")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int,
                        help="number of minibatches to take during each backwardpass preferably multiple of 2")
    parser.add_argument("--verbose", type=int, choices=[1, 2],
                        help="Verbosity mode. 1 = progress bar, 2 = one line per epoch (default 2)")
    parser.add_argument("--classweights", type=bool,
                        help="if True, class weights are added")
    parser.add_argument("--comment",
                        help="Add comments to the log files")

    args = parser.parse_args()
    print("%s selected" % (args.fold_idx))
    fold_idx = args.fold_idx

    if args.seed:  # if random seed is specified
        print("Random seed specified as %d" % (args.seed))
        random_seed = args.seed
    else:
        random_seed = 1
    num_class = 5

    if args.epochs:  # if number of training epochs is specified
        print("Training for %d epochs" % (args.epochs))
        epochs = args.epochs
    else:
        epochs = 200
        print("Training for %d epochs" % (epochs))

    if args.batch_size:  # if batch_size is specified
        print("Training with %d samples per minibatch" % (args.batch_size))
        batch_size = args.batch_size
    else:
        batch_size = 1024
        print("Training with %d minibatches" % (batch_size))

    if args.verbose:
        verbose = args.verbose
        print("Verbosity level %d" % (verbose))
    else:
        verbose = 2
    if args.comment:
        comment = args.comment
    else:
        comment = None

    model_dir = os.path.join(os.getcwd(), '..', 'models')
    fold_dir = os.path.join(os.getcwd(), '..', 'data')
    log_dir = os.path.join(os.getcwd(), '..', 'logs')
    log_name = 'fold_' + str(fold_idx)

    if not os.path.exists(os.path.join(model_dir, log_name)):
        new_dir = (os.path.join(model_dir, log_name))
        print(new_dir)
        os.makedirs(new_dir)

    if not os.path.exists(os.path.join(log_dir, log_name)):
        new_dir = os.path.join(log_dir, log_name)
        print(new_dir)
        os.makedirs(new_dir)

    checkpoint_name = os.path.join(model_dir, log_name, 'weights.hdf5')

    params = {

        'num_classes': num_class,
        'batch_size': batch_size,
        'epochs': epochs,
        'aafoldname': fold_idx,
        'random_seed': random_seed,
        'load_path': load_path,
        'shuffle': True,
        'initial_epoch': initial_epoch,
        'eeg_length': 3000,
        'kernel_size': 16,
        'bias': True,
        'maxnorm': 400000000000.,  ## No maxnorm constraint
        'dropout_rate': 0.45,  # .5
        'dropout_rate_dense': 0.,
        'padding': 'valid',
        'activation_function': 'relu',
        'subsam': 2,
        'trainable': True,
        'lr': .001,  # .0001
        'lr_decay': 0.0  # 1e-5, #1e-5
    }

    current_learning_rate = params['lr']
    inf = glob.glob(os.path.join(fold_dir, 'SC*.csv'))
    df = pd.DataFrame()
    for f in inf:
        df = pd.concat([df, pd.read_csv(f, header=None)])
    trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter(df, fold_idx)
    print(trainX.shape, valX.shape, trainY.shape, valY.shape)
    del df

    print("Data loaded")

    if args.classweights:
        params['class_weight'] = compute_weight(trainY.astype(int), np.unique(trainY.astype(int)))
    else:
        params['class_weight'] = dict(zip(np.r_[0:params['num_classes']], np.ones(params['num_classes'])))

    print('Classwise data in train', Counter(trainY))

    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    trainX = np.expand_dims(trainX, axis=-1)
    valX = np.expand_dims(valX, axis=-1)

    K.clear_session()
    top_model = eegnet(**params)
    x = Flatten()(top_model.output)
    x = Dense(params['num_classes'], activation='softmax', kernel_initializer=initializers.he_normal(seed=random_seed),
              kernel_constraint=max_norm(params['maxnorm']), use_bias=True)(x)

    model = Model(top_model.input, x)
    model.summary()
    if load_path:
        model.load_weights(filepath=load_path, by_name=False)
    model_json = model.to_json()
    with open(os.path.join(model_dir, log_name, 'model.json'), "w") as json_file:
        json_file.write(model_json)

    model.compile(optimizer=opt(lr=params['lr'], epsilon=None, decay=params['lr_decay']),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    ####### Callbacks #######

    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=False, mode='max')

    tensdir = log_dir + "/" + log_name + "/"
    tensbd = TensorBoard(log_dir=tensdir, batch_size=batch_size, write_grads=True, )
    patlogDirectory = log_dir + '/' + log_name + '/'
    trainingCSVdirectory = log_dir + '/' + log_name + '/' + 'training.csv'
    csv_logger = CSVLogger(trainingCSVdirectory)

    ####### Training #########

    try:

        datagen = AudioDataGenerator(
            roll_range=.15,
            samplewise_center=True,
            samplewise_std_normalization=True,
        )

        valgen = AudioDataGenerator(

            samplewise_center=True,
            samplewise_std_normalization=True,
        )

        model.fit_generator(
            datagen.flow(trainX, trainY, batch_size=params['batch_size'], shuffle=True, seed=params['random_seed']),
            steps_per_epoch=len(trainX) // params['batch_size'],
            epochs=params['epochs'],
            validation_data=valgen.flow(valX, valY, batch_size=params['batch_size'],
                                        seed=params['random_seed']),
            callbacks=[modelcheckpnt, csv_logger, tensbd, lrate],
            class_weight=params['class_weight']
            )
        model.save_weights(checkpoint_name)
    except:
        raise
