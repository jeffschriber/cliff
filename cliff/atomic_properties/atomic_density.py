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
import numpy as np
import operator

# Set logger
logger = logging.getLogger(__name__)

class AtomicDensity:
    'AtomicDensity class. Predicts atomic populations and valence widths.'

    def __init__(self,options):
        self.descr_train = []
        self.target_train = []
        # With environment
        self.descr_env_train = []
        self.target_env_train = []
        self.baseline_env_train = []
        # kernel ridge regression
        self.alpha_train = None
        self.alpha_env_train = None
        logger.setLevel(options.logger_level)
        self.max_neighbors     = options.atomicdensity_max_neighbors
        self.max_neighbors_env = options.atomicdensity_max_neighbors_env
        self.krr_sigma = options.atomicdensity_krr_sigma
        self.krr_lambda = options.atomicdensity_krr_lambda
        self.krr_sigma_env = options.atomicdensity_krr_sigma_env
        self.krr_gamma_env = options.atomicdensity_krr_gamma_env
        self.training_file = options.atomicdensity_training
        self.training_env_file = options.atomicdensity_training_env
        self.use_ref_density = options.atomicdensity_ref_adens
        self.refpath = options.atomicdensity_refpath

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

    def load_ml_env(self):
        logger.info(
            "Reading atomic-density refinement training from %s" % training_file)
        with open(self.training_env_file, 'rb') as f:
            self.descr_env_train, self.baseline_env_train, \
                self.alpha_env_train = pickle.load(f)

    def train_ml_env(self):
        '''Train machine learning model embedded in environment.'''
        size_training = len(self.target_env_train)
        if len(self.descr_env_train) == 0:
            print("No molecule in the training set.")
            logger.error("No molecule in the training set.")
            exit(1)
        logger.info("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
            % (size_training, size_training, 8*size_training**2/1e9))
        pairwise_dists = squareform(pdist(self.descr_env_train, 'cityblock'))
        # Baseline predictions require nonzero self.alpha_train
        if self.alpha_train is None:
            raise ValueError("Can't learn refinement without baseline prediction")
        baseline_dists = squareform(pdist(self.baseline_env_train, 'cityblock'))
        kmat = scipy.exp(- pairwise_dists / self.krr_sigma_env
                        - baseline_dists / self.krr_gamma_env)
        kmat += self.krr_lambda*np.identity(len(self.target_env_train))
        self.alpha_env_train = np.linalg.solve(kmat,self.target_env_train)
        logger.info("training finished.")
        return None

    def predict_mol_env(self, _system, _env):
        '''Predict coefficients within environment given descriptors.'''
        _system.build_coulmat_env(_env, self.max_neighbors_env)
        pairwise_dists = cdist(_system.coulmat_env, self.descr_env_train,
            'cityblock')
        # Baseline predictions require nonzero self.alpha_train
        if self.alpha_train is None:
            raise ValueError("Can't learn refinement without baseline prediction")
        self.predict_mol(_system)
        baseline_sys = [[a,v] for a,v in
                    zip(_system.populations, _system.valence_widths)]
        baseline_dists = cdist(baseline_sys, self.baseline_env_train, 'cityblock')
        kmat = scipy.exp(- pairwise_dists / self.krr_sigma_env
                        - baseline_dists / self.krr_gamma_env)
        pred = np.dot(kmat,self.alpha_env_train)
        _system.populations += pred.T[0]
        _system.valence_widths += pred.T[1]
        logger.info("Prediction (refinement): %s" % _system.populations)
        return None

    def add_mol_in_env_to_training(self, new_system, env, populations, valwidths):
        '''Add molecule embedded in environment to training set with given
        populations and valence widths.'''
        new_system.build_coulmat_env(env, self.max_neighbors_env)
        if len(valwidths) != len(new_system.coulmat_env):
            raise ValueError("Inconsistency in training data (%d vs %d)" % (
                    len(valwidths), len(new_system.coulmat_env)))
        self.descr_env_train += new_system.coulmat_env
        if self.alpha_train is None:
            raise ValueError("Can't learn refinement without baseline prediction")
        self.predict_mol(new_system)
        self.baseline_env_train += [[a,v] for a,v in
                    zip(new_system.populations, new_system.valence_widths)]
        # Subtract out molecular prediction from target
        self.target_env_train += [[a-a0,v-v0] for a,a0,v,v0 in
                zip(populations,new_system.populations,valwidths,new_system.valence_widths)]
        logger.info("Added file to env training set: %s" % new_system)
        return None
