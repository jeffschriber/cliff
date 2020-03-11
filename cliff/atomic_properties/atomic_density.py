#!/usr/bin/env python

from cliff.helpers.system import System
import cliff.helpers.utils as utils
import cliff.helpers.constants as constants
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import logging
import pickle
import time
import numpy as np
import operator
import os
import qml
from qml.representations import get_slatm_mbtypes

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]
# Set logger
logger = logging.getLogger(__name__)
fh = logging.FileHandler('output.log')
logger.addHandler(fh)


class AtomicDensity:
    'AtomicDensity class. Predicts atomic populations and valence widths.'

    def __init__(self,options):
        self.descr_train  = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.target_train = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
        self.alpha_train = {'H':None, 'C':None, 'O':None, 'N':None, 'S':None, 'Cl':None, 'F':None, 'Br':None}
        # kernel ridge regression
        logger.setLevel(options.logger_level)
        self.max_neighbors = options.atomicdensity_max_neighbors
        self.krr_sigma = options.atomicdensity_krr_sigma
        self.krr_lambda = options.atomicdensity_krr_lambda
        self.training_file = options.atomicdensity_training
        self.mbtypes = None
        self.qml_mols = []
        self.kernel = 'laplacian'

        self.use_ref_density = options.atomicdensity_ref_adens
        self.refpath = options.atomicdensity_refpath

        # A dict of element types for training/predicting
        self.ele_train = {} 

        if options.test_mode:
            self.refpath = testpath + self.refpath

    def load_ml(self):
        logger.info(
            "    Loading atomic-density training from %s" % self.training_file)

        if self.use_ref_density:
            return
        
        with open(self.training_file, 'rb') as f:
            #self.descr_train, self.alpha_train = pickle.load(f)
            self.descr_train, self.alpha_train = pickle.load(f, encoding='latin1')

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
        #print("Training finished")
        logger.info("training finished.")
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
            # loop over elements
            #for ele in self.alpha_train.keys(): 
            #    if self.alpha_train[ele] is not None:

            # allocate the result
            _system.populations = np.zeros(_system.num_atoms)
            _system.valence_widths = np.zeros(_system.num_atoms)
            
            _system.build_slatm(self.mbtypes) # pass xyz here?

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
                            _system.populations[i] = pred.T[0][i]
                            _system.valence_widths[i] = pred.T[1][i]

                        

#            # Build element-specific coulomb matrices
#            cmat = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
#            idx = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[], 'Br':[]}
#            for i in range(_system.num_atoms):
#                ele = _system.elements[i]
#                if self.alpha_train[ele] is not None:
#                   # _system.build_coulomb_matrices(self.max_neighbors, ele)
#                    coul_mat, reorder_atoms = utils.build_coulomb_matrix(
#                        _system.coords, _system.elements, i, self.max_neighbors)
#                    _system.atom_reorder.append(reorder_atoms)
#                    _system.coulomb_mat.append(coul_mat)
#                    cmat[ele].append(coul_mat)
#                    idx[ele].append(i)
#    
#            # Predict pops/widths, same element types done simultaneously
#            for ele in cmat.keys():
#                if self.alpha_train[ele] is not None:
#                    pairwise_dists = cdist(cmat[ele], self.descr_train[ele],
#                        'cityblock')
#                    kmat = np.exp(- pairwise_dists / self.krr_sigma )
#                    pred = np.dot(kmat,self.alpha_train[ele])
#
#                    for i in range(len(idx[ele])):
#                        _system.populations[idx[ele][i]] = pred.T[0][i]
#                        _system.valence_widths[idx[ele][i]] = pred.T[1][i]
#                       # _system.populations, _system.valence_widths = pred.T[0], pred.T[1]

       # print("    Time spent predicting valence-widths and populations: %8.3f s" % (time.time() - t1))
        return None

    def add_mol_to_training(self, new_system, populations, valwidths, atom=None):

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
        mol.generate_slatm(self.mbtypes, local=True)

        #self.qml_filter_ele = []
        #if atom is None:
        #    for i in range(mol.natoms):
        #        self.qml_filter_ele.append(1)
        #else:
        #    for i in range(mol.natoms):
        #        if str(mol.atomtypes[i]) == atom:   
        #            self.qml_filter_ele.append(1)
        #        else :
        #            self.qml_filter_ele.append(0)

        natom = 0
        for i in range(len(new_system.elements)):
            ele = new_system.elements[i]
            if (ele == atom) or atom is None:
                natom += 1 
                # reference pops/widths for element i
                epop = [populations[i],valwidths[i]] 
                self.target_train[ele].append(epop)
                self.descr_train[ele].append(mol.representation[i])

                if len(self.descr_train[ele]) != len(self.target_train[ele]):
                    print(len(self.descr_train[ele]))
                    print(len(self.target_train[ele]))
                    print(self.descr_train[ele])
                    print(self.target_train[ele])
                    print("Inconsistency in training data")
                    raise ValueError("Inconsistency in training data")


   # def add_mol_to_training(self, new_system, populations, valwidths, atom=None):
   #     'Add molecule to training set with given populations and valence widths.'

   #     if atom is None:
   #         for i in new_system.elements:
   #             self.ele_train[i] = 1
   #     else:
   #         self.ele_train[atom] = 1

   #     epop = {}
   #     natom = 0
   #     for i in range(len(new_system.elements)):
   #         ele = new_system.elements[i]
   #         if (ele == atom) or atom is None:
   #             natom += 1
   #             #new_system.build_coulomb_matrices(self.max_neighbors, ele)
   #             coul_mat, reorder_atoms = utils.build_coulomb_matrix(
   #                 new_system.coords, new_system.elements, i, self.max_neighbors)
   #             new_system.atom_reorder.append(reorder_atoms)
   #             new_system.coulomb_mat.append(coul_mat)

   #             #epop[ele].append([populations[i],valwidths[i]]) 
   #             epop = [populations[i],valwidths[i]] 

   #             self.descr_train[ele].append(coul_mat)
   #             #self.descr_train[ele].append(new_system.coulomb_mat)
   #             #self.target_train[ele].append(epop)
   #             self.target_train[ele] += [epop]

   #             if len(self.descr_train[ele]) != len(self.target_train[ele]):
   #                 print(len(self.descr_train[ele]))
   #                 print(len(self.target_train[ele]))
   #                 print(self.descr_train[ele])
   #                 print(self.target_train[ele])
   #                 print("Inconsistency in training data")
   #                 raise ValueError("Inconsistency in training data")
   #             #self.descr_train += new_system.coulomb_mat
   #             #self.target_train += [[a,v] for a,v in zip(populations,valwidths)]

        logger.info("Added file to training set: %s" % new_system)
        return natom
