from __future__ import print_function, absolute_import, division
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
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

    if args.loadmodel:  # If a previously trained model is loaded for retraining
        load_path = args.loadmodel  #### path to model to be loaded

        idx = load_path.find("weights")
        initial_epoch = int(load_path[idx + 8:idx + 8 + 4])

        print("%s model loaded\nInitial epoch is %d" % (args.loadmodel, initial_epoch))
    else:
        print("no model specified, using initializer to initialize weights")
        initial_epoch = 0
        load_path = False

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

    params = {

        'num_classes': num_class,
        'batch_size': batch_size,
        'epochs': epochs,
        'aafoldname': fold_idx,
        'random_seed': random_seed,
        'load_path': load_path,
        'shuffle': False,
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
        'trainable': False,
        'lr': .001,  # .0001
        'lr_decay': 0.0  # 1e-5, #1e-5
    }

    current_learning_rate = params['lr']
    inf = glob.glob(os.path.join(fold_dir, 'SC*.csv'))
    df = pd.DataFrame()
    for f in inf:
        df = pd.concat([df, pd.read_csv(f, header=None)])

    total_valY = []
    total_ypred = []
    for fold_idx in range(20):
        log_name = 'fold_' + str(fold_idx)
        checkpoint_name = os.path.join(model_dir, log_name, 'weights.hdf5')
        trainX, valX, trainY, valY, pat_train, pat_val = patientSplitter(df, fold_idx)
        print(trainX.shape, valX.shape, trainY.shape, valY.shape)

        print("Data loaded")

        if args.classweights:
            params['class_weight'] = compute_weight(trainY.astype(int), np.unique(trainY.astype(int)))
        else:
            params['class_weight'] = dict(zip(np.r_[0:params['num_classes']], np.ones(params['num_classes'])))

        print('Classwise data in train', Counter(trainY))

        valY = to_categorical(valY)
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

        valgen = AudioDataGenerator(

            samplewise_center=True,
            samplewise_std_normalization=True,
        )
        print(checkpoint_name)
        model.load_weights(checkpoint_name)
        y_pred = model.predict_generator(valgen.flow(valX, None, batch_size=64, seed=1, shuffle=False))
        y_pred = np.argmax(y_pred, axis=1)
        valY = np.argmax(valY, axis=1)
        acc = np.mean(valY == y_pred)
        print('acc', acc)
        mf1 = f1_score(valY, y_pred, average="macro")
        print('mf1', mf1)
        ck = cohen_kappa_score(valY, y_pred)
        print('ck', ck)
        cm = confusion_matrix(valY, y_pred)
        print('cm', cm)

        total_valY.extend(valY)
        total_ypred.extend(y_pred)

    print("Overall prediction performance\n")
    y_true = np.asarray(total_valY)
    y_pred = np.asarray(total_ypred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    ck = cohen_kappa_score(y_true, y_pred)
    print(
        "n={}, acc={:.3f}, f1={:.3f}, ck={:.3f}".format(
            n_examples, acc, mf1, ck
        )
    )
    print(cm)

