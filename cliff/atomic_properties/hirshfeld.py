#!/usr/bin/env python
#
# Hirshfeld class. Predict Hirshfeld ratios.
#
# Tristan Bereau (2017)

from cliff.helpers.system import System
import cliff.helpers.utils
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import logging
import pickle
import numpy as np
import os
import time

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]
# Set logger
logger = logging.getLogger(__name__)

class Hirshfeld:
    'Hirshfeld class. Predicts Hirshfeld ratios.'

    def __init__(self, options):
        self.descr_train = []
        self.target_train = []
        # kernel ridge regression
        self.alpha_train = None
        # support vector regression
        self.clf = None
        logger.setLevel(options.logger_level)
        self.max_neighbors = options.hirsh_max_neighbors
        self.krr_kernel = options.hirsh_krr_kernel
        self.krr_sigma  = options.hirsh_krr_sigma
        self.krr_lambda = options.hirsh_krr_lambda

        self.from_file = options.hirsh_file_read
        self.filepath  = options.hirsh_filepath

        if options.test_mode:
            self.filepath = testpath + self.filepath
        
        self.training_file = options.hirsh_training


    def load_ml(self):

        if self.from_file:
            return

        logger.info(
            "Reading Hirshfeld training from %s" % self.training_file)
        with open(self.training_file, 'rb') as f:
            self.descr_train, self.alpha_train = pickle.load(f, encoding='bytes')

    def train_ml(self, ml_method):
        '''Train machine learning model.'''
        size_training = len(self.target_train)
        if len(self.descr_train) == 0:
            logger.error("No molecule in the training set.")
            exit(1)
        if ml_method == "krr":
            logger.info("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
                % (size_training, size_training, 8*size_training**2/1e9))
            print("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
                % (size_training, size_training, 8*size_training**2/1e9))
            if self.krr_kernel == 'gaussian':
                pairwise_dists = squareform(pdist(self.descr_train, 'euclidean'))
                kmat = scipy.exp(- pairwise_dists**2 / (2.*self.krr_sigma**2) )
            elif self.krr_kernel == 'laplacian':
                pairwise_dists = squareform(pdist(self.descr_train, 'cityblock'))
                kmat = scipy.exp(- pairwise_dists / self.krr_sigma )
            else:
                print("Kernel",self.krr_kernel,"not implemented.")
            kmat += self.krr_lambda*np.identity(len(self.target_train))
            self.alpha_train = np.linalg.solve(kmat,self.target_train)
        else:
            logger.error("unknown ML method %s" % ml_method)
            exit(1)
        logger.info("training finished.")
        return None

    def predict_mol(self, _system, ml_method):
        '''Predict coefficients given  descriptors.'''
        t1 = time.time()
        _system.build_coulomb_matrices(self.max_neighbors)
        if self.from_file:
            h_ratios = []
            for hfile in _system.xyz:
                hfile = self.filepath + hfile.split('/')[-1]
                with open(hfile,'r') as infile:
                    for line in infile:
                        line = line.split()
                        if len(line) == 6:
                            h_ratios.append(float(line[4])) 

            _system.hirshfeld_ratios = h_ratios

        elif ml_method == "krr":
            if self.krr_kernel == 'gaussian':
                pairwise_dists = cdist(_system.coulomb_mat, self.descr_train,
                    'euclidean')
                kmat = scipy.exp(- pairwise_dists**2 / (2.*self.krr_sigma**2) )
            elif self.krr_kernel == 'laplacian':
                pairwise_dists = cdist(_system.coulomb_mat, self.descr_train,
                    'cityblock')
                kmat = scipy.exp(- pairwise_dists / self.krr_sigma )
            else:
                logger.error("Kernel %s not implemented" % self.krr_kernel)
                exit(1)
            _system.hirshfeld_ratios = np.dot(kmat,self.alpha_train)
        else:
            logger.error("unknown ML method %s" % ml_method)
            exit(1)
        logger.info("Prediction: %s" % _system.hirshfeld_ratios)

        print("    Time spent predicting Hirshfeld ratios:               %8.3f s" % (time.time()-t1))
        return None

    def add_mol_to_training(self, new_system):
        'Add molecule to training set'
        new_system.build_coulomb_matrices(self.max_neighbors)
        if len(new_system.hirshfeld_ref) != len(new_system.coulomb_mat):
            logger.error("Inconcistency in training data for %s" % new_system)
            exit(1)
        self.descr_train += new_system.coulomb_mat
        self.target_train += [i for i in new_system.hirshfeld_ref]
        logger.info("Added file to training set: %s" % new_system)
        return None
