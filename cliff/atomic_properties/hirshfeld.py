#!/usr/bin/env python
#
# Hirshfeld class. Predict Hirshfeld ratios.
#
# Tristan Bereau (2017)

from cliff.helpers.system import System
import cliff.helpers.utils
import cliff.helpers.constants as constants
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import logging
import pickle
import numpy as np
import os
import time
import glob
import qml
from qml.representations import get_slatm_mbtypes

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]
# Set logger
logger = logging.getLogger(__name__)
fh = logging.FileHandler('output.log')
logger.addHandler(fh)

class Hirshfeld:
    'Hirshfeld class. Predicts Hirshfeld ratios.'

    def __init__(self, options):
        self.descr_train  = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.target_train = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.alpha_train = {'H':None, 'C':None, 'O':None, 'N':None, 'S':None, 'Cl':None, 'F':None, 'Br':None}
        # support vector regression
        self.clf = None
        logger.setLevel(options.logger_level)
        self.cutoff = options.hirsh_cutoff
        self.krr_kernel = options.hirsh_krr_kernel
        self.krr_sigma  = options.hirsh_krr_sigma
        self.krr_lambda = options.hirsh_krr_lambda
        self.mbtypes = None
        self.qml_mols = []
        self.kernel = 'laplacian'

        self.from_file = options.hirsh_file_read
        self.filepath  = options.hirsh_filepath

        # A dict of element types for training/predicting
        self.ele_train = {} 
        if options.test_mode:
            self.filepath = testpath + self.filepath
        
        self.training_dir = options.hirsh_training


    def load_ml(self):

        if self.from_file:
            return

        logger.info(
            "    Loading Hirshfeld training from %s" % self.training_dir)
        hirsh_models = glob.glob(self.training_dir + '/*.pkl') 
        for model in hirsh_models:
            with open(model, 'rb') as f:
                #self.descr_train, self.alpha_train = pickle.load(f)
                d_train,a_train, self.mbtypes = pickle.load(f, encoding='latin1')

                for ele in self.descr_train.keys():
                    if ele in d_train.keys() and len(d_train[ele]) > 0:
                        self.descr_train[ele] = d_train[ele]
                        self.alpha_train[ele] = a_train[ele]
        return None

    def save_ml(self, save_file):
        '''save the model'''

        with open (save_file, 'wb') as f:
            pickle.dump([self.descr_train,self.alpha_train,self.mbtypes],f,protocol=2)

        return None

    def train_ml(self):
        '''Train machine learning model.'''

        if len(self.descr_train) == 0:
            print("No molecule in the training set.")
            logger.error("No molecule in the training set.")
            exit(1)
        for ele in self.descr_train.keys():
            size_training = len(self.target_train[ele])
        
            # We've already got the descriptors
            if len(self.descr_train[ele]) > 0:

                logger.info("Training set size: %d atoms" % size_training)                
                pairwise_dists = squareform(pdist(self.descr_train[ele], 'cityblock'))
                kmat = np.exp(- pairwise_dists / self.krr_sigma )
                kmat += self.krr_lambda*np.identity(len(self.target_train[ele]))
                self.alpha_train[ele] = np.linalg.solve(kmat,self.target_train[ele])

#        size_training = len(self.target_train)
#        if len(self.descr_train) == 0:
#            logger.error("No molecule in the training set.")
#            exit(1)
#
#        logger.info("   Building kernel matrix of size (%d,%d); %7.4f Gbytes" \
#            % (size_training, size_training, 8*size_training**2/1e9))
#        print("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
#            % (size_training, size_training, 8*size_training**2/1e9))
#
#        if self.krr_kernel == 'gaussian':
#            pairwise_dists = squareform(pdist(self.descr_train, 'euclidean'))
#            kmat = np.exp(- pairwise_dists**2 / (2.*self.krr_sigma**2) )
#        elif self.krr_kernel == 'laplacian':
#            pairwise_dists = squareform(pdist(self.descr_train, 'cityblock'))
#            kmat = np.exp(- pairwise_dists / self.krr_sigma )
#        else:
#            print("Kernel",self.krr_kernel,"not implemented.")

        logger.info("training finished.")
        return None

    def predict_mol(self, _system):
        '''Predict coefficients given  descriptors.'''
        t1 = time.time()

        #_system.build_coulomb_matrices(self.max_neighbors)
        if self.from_file:
            h_ratios = []
            for hfile in _system.xyz:
                hfile = self.filepath + hfile.split('/')[-1].strip('.xyz') + '-h.txt'
                with open(hfile,'r') as infile:
                    for line in infile:
                        line = line.split()
                        # Some reference files include coordinates
                        if len(line) == 6:
                            h_ratios.append(float(line[4])) 
                        else:
                            h_ratios.append(float(line[0]))

            _system.hirshfeld_ratios = h_ratios

        else:

            _system.hirshfeld_ratios = np.zeros(_system.num_atoms)
            _system.build_slatm(self.mbtypes, self.cutoff) # pass xyz here?

            prefactor = constants.ml_prefactor[self.kernel]
            power = constants.ml_power[self.kernel]
            for ele in self.alpha_train.keys():
                if self.alpha_train[ele] is not None:
                    pairwise_dists = cdist(_system.slatm, self.descr_train[ele], constants.ml_metric[self.kernel])
                    kmat = np.exp(- pairwise_dists / (prefactor*self.krr_sigma**power) )
                    pred = np.dot(kmat,self.alpha_train[ele])
                    for i in range(_system.num_atoms):
                        if _system.elements[i] == ele:
                            _system.hirshfeld_ratios[i] = pred[i]

        #    if self.krr_kernel == 'gaussian':
        #        pairwise_dists = cdist(_system.coulomb_mat, self.descr_train,
        #            'euclidean')
        #        kmat = np.exp(- pairwise_dists**2 / (2.*self.krr_sigma**2) )
        #    elif self.krr_kernel == 'laplacian':
        #        pairwise_dists = cdist(_system.coulomb_mat, self.descr_train,
        #            'cityblock')
        #        kmat = np.exp(- pairwise_dists / self.krr_sigma )
        #    else:
        #        logger.error("Kernel %s not implemented" % self.krr_kernel)
        #        exit(1)
        #    _system.hirshfeld_ratios = np.dot(kmat,self.alpha_train)

       # print("    Time spent predicting Hirshfeld ratios:               %8.3f s" % (time.time()-t1))
#            xyz = _system.xyz[0].split('/')[-1].strip('.xyz')
#            reffile = self.filepath + xyz + '-h.txt'
#            with open(reffile,'w') as ref:
#                for hr in _system.hirshfeld_ratios:
#                    ref.write("%10.8f \n" % hr)
        return None

    def add_mol_to_training(self, new_system, ref_ratios,atom = None):
        'Add molecule to training set'

        if self.mbtypes is None:
            raise ValueError("Missing MBTypes")

        mol = None
        # Init the molecule in qml
        if new_system.xyz[0] is None:
            if xyz is not None:
                mol = qml.Compound(xyz)
            else:
                raise ValueError("Missing xyz file")
        else:
            mol = qml.Compound(new_system.xyz[0])  

        self.qml_mols.append(mol)
        # build slatm representation
        mol.generate_slatm(self.mbtypes, rcut = self.cutoff, local=True)

        natom = 0
        for i in range(len(new_system.elements)):
            ele = new_system.elements[i]
            if (ele == atom) or atom is None:
                natom += 1 
                # reference pops/widths for element i
                hr = ref_ratios[i] 
                self.target_train[ele].append(hr)
                self.descr_train[ele].append(mol.representation[i])

                if len(self.descr_train[ele]) != len(self.target_train[ele]):
                    print(len(self.descr_train[ele]))
                    print(len(self.target_train[ele]))
                    print(self.descr_train[ele])
                    print(self.target_train[ele])
                    print("Inconsistency in training data")
                    raise ValueError("Inconsistency in training data")
        #self.descr_train += new_system.coulomb_mat
        #self.target_train += [i for i in new_system.hirshfeld_ref]

        logger.info("Added file to training set: %s" % new_system)
        return natom
