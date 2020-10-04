#!/usr/bin/env python

from cliff.helpers.system import System
import cliff.helpers.utils as utils
import cliff.helpers.constants as constants
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import logging
import pickle as pkl
import time
import numpy as np
import operator
import os
import qml
from qml.representations import get_slatm_mbtypes
import glob

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

class AtomicDensity:
    'AtomicDensity class. Predicts atomic populations and valence widths.'

    def __init__(self,options):
        name = options.name
        # Set logger
        self.logger = options.logger

        self.name = name
        self.descr_train  = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.target_train = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.alpha_train = {'H':None, 'C':None, 'O':None, 'N':None, 'S':None, 'Cl':None, 'F':None, 'Br':None}
        # kernel ridge regression
        self.krr_sigma = options.atomicdensity_krr_sigma
        self.krr_lambda = options.atomicdensity_krr_lambda
        self.training_file = options.atomicdensity_training
        self.mbtypes = None
        self.qml_mols = []
        self.kernel = 'laplacian'
        self.training_dir = options.atomicdensity_training

        self.use_ref_density = options.atomicdensity_ref_adens
        self.refpath = options.atomicdensity_refpath

        self.cutoff = options.atomicdensity_cutoff

        self.save_to_disk = options.atomicdensity_save_to_disk
        self.save_path = options.atomicdensity_save_path

        # A dict of element types for training/predicting
        self.ele_train = {} 

        if options.test_mode:
            self.refpath = testpath + self.refpath

    def load_ml(self):
        self.logger.info(
            "    Loading atomic-density training from %s" % self.training_dir)
        if self.use_ref_density:
            return

        adens_models = glob.glob(self.training_dir + '/*.pkl') 
        for model in adens_models:
            with open(model, 'rb') as f:
                d_train,a_train, self.mbtypes = pkl.load(f)
                for ele in self.descr_train.keys():
                    if ele in d_train.keys() and len(d_train[ele]) > 0:
                        self.descr_train[ele] = d_train[ele]
                        self.alpha_train[ele] = a_train[ele]
        return None


    def save_ml(self, save_file):
        '''save the model'''

        with open (save_file, 'wb') as f:
            pkl.dump([self.descr_train,self.alpha_train,self.mbtypes],f,protocol=2)

        return None

    def train_ml(self):
        '''Train machine learning model.'''
        if len(self.descr_train) == 0:
            print("No molecule in the training set.")
            self.logger.error("No molecule in the training set.")
            exit(1)

        for ele in self.descr_train.keys():
            size_training = len(self.target_train[ele])
        
            # We've already got the descriptors
            if len(self.descr_train[ele]) > 0:

                self.logger.info("Training set size: %d atoms" % size_training)                
                pairwise_dists = squareform(pdist(self.descr_train[ele], 'cityblock'))

                kmat = np.exp(- pairwise_dists / self.krr_sigma )
                kmat += self.krr_lambda*np.identity(len(self.target_train[ele]))
                self.alpha_train[ele] = np.linalg.solve(kmat,self.target_train[ele])
        #print("Training finished")
        self.logger.info("Training finished.")
        return None

    def predict_mol(self, _system):
        '''Predict coefficients given  descriptors.'''
        t1 = time.time()

        if self.use_ref_density:
            xyz = _system.xyz[0].split('/')[-1].strip('.xyz')
            reffile = self.refpath + xyz + '-atmdns.txt'
            vws = []

            with open(reffile, 'r') as f:
                for line in f:
                    line = line.split()
                    vws.append(float(line[0]))
                
            _system.valence_widths = vws

        else:
            # loop over elements
            #for ele in self.alpha_train.keys(): 
            #    if self.alpha_train[ele] is not None:

            # allocate the result
            _system.valence_widths = np.zeros(_system.num_atoms)
            
            _system.build_slatm(self.mbtypes, self.cutoff) # pass xyz here?

            prefactor = constants.ml_prefactor[self.kernel]
            power = constants.ml_power[self.kernel]

            for ele in self.alpha_train.keys():
                if self.alpha_train[ele] is not None:

                    pairwise_dists = cdist(_system.slatm, self.descr_train[ele], constants.ml_metric[self.kernel])
                    #kmat = np.exp(- pairwise_dists / self.krr_sigma )
                    kmat = np.exp(- pairwise_dists / (prefactor*self.krr_sigma**power) )
                    pred = np.dot(kmat,self.alpha_train[ele])

                    for i in range(_system.num_atoms):
                        if _system.elements[i] == ele:
                            #_system.populations[i] = pred.T[0][i]
                            #_system.valence_widths[i] = pred.T[1][i]
                            _system.valence_widths[i] = pred.T[i]

            if self.save_to_disk:
                xyz = _system.xyz[0].split('/')[-1].strip('.xyz')
                reffile = self.save_path + xyz + '-atmdns.txt'
                with open(reffile,'w') as ref:
                    for vw in _system.valence_widths:
                        ref.write("%10.8f \n" % vw)
            
       # print("    Time spent predicting valence-widths and populations: %8.3f s" % (time.time() - t1))
        return None

    def add_mol_to_training(self, new_system, valwidths, atom=None):

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

        # Only predict the widths
        natom = 0
        for i in range(len(new_system.elements)):
            ele = new_system.elements[i]
            if (ele == atom) or atom is None:
                natom += 1 
                # reference pops/widths for element i
                self.target_train[ele].append(valwidths[i])
                self.descr_train[ele].append(mol.representation[i])

                if len(self.descr_train[ele]) != len(self.target_train[ele]):
                    print(len(self.descr_train[ele]))
                    print(len(self.target_train[ele]))
                    print(self.descr_train[ele])
                    print(self.target_train[ele])
                    print("Inconsistency in training data")
                    raise ValueError("Inconsistency in training data")

        self.logger.info("Added file to training set: %s" % new_system)
        return natom
