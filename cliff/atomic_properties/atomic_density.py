#!/usr/bin/env python
#
# AtomicDensity class. Predicts atomic populations and valence widths.
#
# Tristan Bereau (2016)

from cliff.helpers.system import System
import cliff.helpers.utils
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import logging
import pickle
import time
import numpy as np
import operator
import os

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]
# Set logger
logger = logging.getLogger(__name__)

class AtomicDensity:
    'AtomicDensity class. Predicts atomic populations and valence widths.'

    def __init__(self,options):
        self.descr_train = []
        self.target_train = []
        # kernel ridge regression
        self.alpha_train = None
        logger.setLevel(options.logger_level)
        self.max_neighbors     = options.atomicdensity_max_neighbors
        self.krr_sigma = options.atomicdensity_krr_sigma
        self.krr_lambda = options.atomicdensity_krr_lambda
        self.training_file = options.atomicdensity_training

        self.use_ref_density = options.atomicdensity_ref_adens
        self.refpath = options.atomicdensity_refpath

        if options.test_mode:
            self.refpath = testpath + self.refpath

    def load_ml(self):
        logger.info(
            "Reading atomic-density training from %s" % self.training_file)

        if self.use_ref_density:
            return
        
        with open(self.training_file, 'rb') as f:
            #self.descr_train, self.alpha_train = pickle.load(f)
            self.descr_train, self.alpha_train = pickle.load(f, encoding='latin1')

    def train_ml(self):
        '''Train machine learning model.'''
        size_training = len(self.target_train)
        if len(self.descr_train) == 0:
            print("No molecule in the training set.")
            logger.error("No molecule in the training set.")
            exit(1)
        pairwise_dists = squareform(pdist(self.descr_train, 'cityblock'))
        kmat = scipy.exp(- pairwise_dists / self.krr_sigma )
        kmat += self.krr_lambda*np.identity(len(self.target_train))
        self.alpha_train = np.linalg.solve(kmat,self.target_train)
        logger.info("training finished.")
        print("training finished.")
        return None

    def predict_mol(self, _system):
        '''Predict coefficients given  descriptors.'''
        t1 = time.time()

        if self.use_ref_density:
            xyz = _system.xyz[0].split('/')[-1].strip('.xyz')
            reffile = self.refpath + xyz + '-atmdns.txt'

            pops = []
            vws = []

            with open(reffile, 'r') as f:
                for line in f:
                    line = line.split()
                    pops.append(float(line[0]))
                    vws.append(float(line[1]))

            _system.populations, _system.valence_widths = pops, vws

        else:
            _system.build_coulomb_matrices(self.max_neighbors)
            pairwise_dists = cdist(_system.coulomb_mat, self.descr_train,
                'cityblock')
            kmat = scipy.exp(- pairwise_dists / self.krr_sigma )
            pred = np.dot(kmat,self.alpha_train)
            _system.populations, _system.valence_widths = pred.T[0], pred.T[1]
        logger.info("Prediction: %s" % _system.populations)
        print("    Time spent predicting valence-widths and populations: %8.3f s" % (time.time() - t1))
        return None

    def add_mol_to_training(self, new_system, populations, valwidths):
        'Add molecule to training set with given populations and valence widths.'
        new_system.build_coulomb_matrices(self.max_neighbors)
        if len(valwidths) != len(new_system.coulomb_mat):
            print("Inconsistency in training data")
            raise ValueError("Inconsistency in training data")
        self.descr_train += new_system.coulomb_mat
        self.target_train += [[a,v] for a,v in zip(populations,valwidths)]
        #print("Added file to training set: %s" % new_system)
        logger.info("Added file to training set: %s" % new_system)
        return None
