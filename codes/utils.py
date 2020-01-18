from __future__ import print_function, absolute_import, division
import numpy as np

np.random.seed(1)
from tensorflow import set_random_seed

set_random_seed(1)
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback
import pandas as pd
from sklearn.metrics import confusion_matrix
from modules import *
from keras.callbacks import LearningRateScheduler


def step_decay(global_epoch_counter):
    lrate = .001
    if global_epoch_counter > 10:
        lrate = .001 / 10
        if global_epoch_counter > 20:
            lrate = .001 / 100
    return lrate


lrate = LearningRateScheduler(step_decay)


def standardnormalization(distribution):
    from sklearn.preprocessing import StandardScaler
    data = distribution.flatten('A')
    data = np.expand_dims(data, axis=1)
    scaler = StandardScaler()
    outdata = scaler.fit_transform(data)
    toutdata = outdata.reshape(int(outdata.shape[0] / 3000), 3000)
    return toutdata


def compute_weight(Y, classes):
    num_samples = len(Y)
    n_classes = len(classes)
    Y = Y.astype(int)
    Y = np.expand_dims(Y, axis=1)
    num_bin = np.bincount(Y[:, 0])
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(5)}
    return class_weights


def train_test_split(fold_idx):
    files = [
        'SC4001E0',
        'SC4002E0',
        'SC4011E0',
        'SC4012E0',
        'SC4021E0',
        'SC4022E0',
        'SC4031E0',
        'SC4032E0',
        'SC4041E0',
        'SC4042E0',
        'SC4051E0',
        'SC4052E0',
        'SC4061E0',
        'SC4062E0',
        'SC4071E0',
        'SC4072E0',
        'SC4081E0',
        'SC4082E0',
        'SC4091E0',
        'SC4092E0',
        'SC4101E0',
        'SC4102E0',
        'SC4111E0',
        'SC4112E0',
        'SC4121E0',
        'SC4122E0',
        'SC4131E0',
        'SC4141E0',
        'SC4142E0',
        'SC4151E0',
        'SC4152E0',
        'SC4161E0',
        'SC4162E0',
        'SC4171E0',
        'SC4172E0',
        'SC4181E0',
        'SC4182E0',
        'SC4191E0',
        'SC4192E0'
    ]
    subject_files = []
    for f in files:
        if fold_idx < 10 and f.startswith('SC40{}'.format(fold_idx)):
            subject_files.append(f)
        elif fold_idx > 9 and f.startswith('SC4{}'.format(fold_idx)):
            subject_files.append(f)
    assert subject_files != []
    train_files = list(set(files) - set(subject_files))
    train_files.sort()
    subject_files.sort()
    return train_files, subject_files


def patientSplitter(data, fold_idx):
    tr, te = train_test_split(int(fold_idx))
    print('train: ', tr)
    print('test: ', te)
    trainMask = np.asarray([each in tr for each in data.iloc[:, 3002]])
    testMask = np.asarray([each in tr for each in data.iloc[:, 3002]])

    X_train = data[trainMask].values[:, :3000]
    X_test = data[testMask].values[:, :3000]
    Y_train = data.iloc[:, 3000][trainMask].values
    Y_test = data.iloc[:, 3000][testMask].values
    pat_train = [int(each[2:5]) for each in data.iloc[:, 3002][trainMask].values]
    pat_test = [int(each[2:5]) for each in data.iloc[:, 3002][testMask].values]
    return X_train, X_test, Y_train, Y_test, pat_train, pat_test



class log_metrics(Callback):
    def __init__(self, valX, valY, patID, patlogDirectory, global_epoch_counter, **kwargs):

        self.patID = patID
        super(log_metrics, self).__init__(**kwargs)
        self.patlogDirectory = patlogDirectory
        self.global_epoch_counter = global_epoch_counter
        self.valY = np.argmax(valY, axis=-1)
        # self.valY = np.expand_dims(self.valY, axis=1)
        self.valX = valX - np.mean(valX, keepdims=True)
        self.valX /= (np.std(self.valX, keepdims=True) + K.epsilon())

    def accuracy_score(self, true, predY):
        match = sum(predY == true)
        return match / len(true)

    def calcMetrics(self, predY, mask=None):

        if mask is None:
            mask = np.ones(shape=self.valY.shape).astype(bool)

        true = self.valY[mask]
        predY = predY[mask]
        confMat = confusion_matrix(true, predY)
        match = sum(predY == true)  # total matches

        sens = []
        spec = []
        acc = []
        for each in np.unique(true).astype(int):
            # each = int(each)
            sens.append(confMat[each, each] / sum(confMat[each, :]))
            spec.append((match - confMat[each, each]) / (
                (match - confMat[each, each] + sum(confMat[:, each]) - confMat[each, each])))
            acc.append(match / (match + sum(confMat[:, each] + sum(confMat[each, :] - 2 * confMat[each, each]))))

        return sens, spec, acc

    def on_epoch_end(self, epoch, logs):

        if logs is not None:

            predY = self.model.predict(self.valX, verbose=0)
            predY = np.argmax(predY, axis=-1)

            sens, spec, acc = self.calcMetrics(predY)

            for sens_, spec_, acc_, class_ in zip(sens, spec, acc, set(self.valY)):
                logs["Class%d-Sens" % class_] = sens_
                logs["Class%d-Spec" % class_] = spec_
                logs["Class%d-Acc" % class_] = acc_

            patAcc = []
            for pat in np.unique(self.patID).astype(int):
                mask = self.patID[:, 0] == pat
                acc = self.accuracy_score(self.valY[mask], predY[mask])
                patAcc.append(acc)
            logs["patAcc"] = np.mean(patAcc)

            sens, spec, acc = self.calcMetrics(predY, self.patID[:, 0] <= 39)
            logs['SC-Sens'] = np.mean(sens)
            logs['SC-Spec'] = np.mean(spec)
            logs['SC-Acc'] = np.mean(acc)

            if sum(self.patID[:, 0] > 39):
                sens, spec, acc = self.calcMetrics(predY, self.patID[:, 0] > 39)
                logs['ST-Sens'] = np.mean(sens)
                logs['ST-Spec'] = np.mean(spec)
                logs['ST-Acc'] = np.mean(acc)

            self.global_epoch_counter = self.global_epoch_counter + 1

            ##############
            lr = self.model.optimizer.lr
            if self.model.optimizer.initial_decay > 0:
                lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
                                                                      K.dtype(self.model.optimizer.decay))))
            t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (
                        1. - K.pow(self.model.optimizer.beta_1, t)))
            logs['lr'] = np.array(float(K.get_value(lr_t)))
