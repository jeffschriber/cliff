#!/usr/bin/env python
#
# Multipoles_ml_bset class. Predict multipole parameters from ML.
# No local axis system. Instead, basis set expansion along the pairwise vectors.

# Tristan Bereau (2016)

#from electrostatics import Electrostatics
from cliff.helpers.system import System
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np
import logging
import pickle
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
import math
import os
import copy
import time
import glob

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]
# qml
import qml
from qml.representations import get_slatm_mbtypes

# Set logger
logger = logging.getLogger(__name__)

class MultipoleMLBSet:
    '''
    MultipoleMLBSet class. Predicts multipoles from machine learning.
    No local axis system. Instead, basis set expansion along the pairwise vectors.
    '''

    def __init__(self, options, descriptor="coulombmatrix"):
        logger.setLevel(options.logger_level)
        self.multipoles = None
        self.descr_train  = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[]}
        self.target_train = {'H':[], 'C':[], 'O':[], 'N':[], 'S':[], 'Cl':[], 'F':[]}
        self.max_neighbors = options.multipole_max_neighbors
        # support vector regression
        self.clf = None
        # alpha_train has size 1,3,9
        self.max_coeffs = [1, 3, 9]
        self.offset_mtp = [0, 1, 4]
        self.alpha_train = {'H':None, 'C':None, 'O':None, 'N':None, 'S':None, 'Cl':None, 'F':None}
        self.kernel     = options.multipole_kernel
        self.krr_sigma  = options.multipole_krr_sigma
        self.krr_lambda = options.multipole_krr_lambda
        # Normalization of the target data - mean and std for each MTP component
        self.norm_tgt_mean = {'H':np.zeros((3)),'C':np.zeros((3)),'O':np.zeros((3)), 'N':np.zeros((3)), 'S':np.zeros((3)), 'Cl':np.zeros((3)), 'F':np.zeros((3))}
        self.norm_tgt_std  = {'H':np.ones((3)), 'C':np.ones((3)), 'O':np.ones((3)), 'N':np.ones((3)), 'S':np.ones((3)), 'Cl':np.ones((1)), 'F':np.zeros((3))}
        self.num_mols_train = {'H':0, 'C':0, 'O':0, 'N':0, 'S':0, 'Cl':0, 'F':0}
        # Descriptor either coulombmatrix or slatm
        self.descriptor = descriptor
        # Set of qml mols. Used for SLATM
        self.qml_mols = []
        self.qml_filter_ele = []
        self.mbtypes = None
        self.ref_mtp = options.multipole_ref_mtp

        self.ref_path = ""
        if self.ref_mtp:
            self.ref_path = options.multipole_ref_path
        if options.test_mode:
            self.ref_path = testpath + self.ref_path

        self.correct_charge = options.multipole_correct_charge

        ## Load the models on init
        if not self.ref_mtp:
            print("    Loading multipole models")
            mtp_s = time.time()
            mtp_models = glob.glob(options.multipole_training + '/*.pkl') 
            for model in mtp_models:
                self.load_ml(model)
            mtp_e = time.time() 
            print("    Loaded {} multipole models in:\n\t\t {}".format(len(mtp_models), options.multipole_training))
            print("    Took %7.4f s to load multipole models" % (mtp_e - mtp_s))



    def load_ml(self, load_file=None):
        '''Load machine learning model'''
        # Try many atoms and see which atoms we find
        if load_file != None:
            logger.info(
                    "Reading multipole training from %s" % load_file)
            with open(load_file, 'rb') as f:
                descr_train_at, alpha_train, norm_tgt_mean, \
                    norm_tgt_std, mbtypes = pickle.load(f)
                    #norm_tgt_std, mbtypes = pickle.load(f, encoding='latin1') #try for old pickles
                for e in self.descr_train.keys():
                    if e in descr_train_at.keys() and len(descr_train_at[e]) > 0:
                        # Update
                        self.descr_train[e] = descr_train_at[e]
                        self.alpha_train[e] = alpha_train[e]
                        self.norm_tgt_mean[e] = norm_tgt_mean[e]
                        self.norm_tgt_std[e] = norm_tgt_std[e]
                self.mbtypes = mbtypes
        else:
            logger.error("Missing load file name")
            exit(1)
        return None

    def save_ml(self, save_file):
        '''Save machine learning model'''
        logger.info("Saving multipole machine learning model to %s" %
            save_file)
        with open(save_file, 'wb') as f:
            pickle.dump([self.descr_train, self.alpha_train,
                         self.norm_tgt_mean, self.norm_tgt_std, self.mbtypes], f, protocol=2)
        return None

    def train_mol(self):
        '''Train machine learning model of multipole rank mtp_rank and
        basis set expansion coefficient coeff.'''
        # SLATM: First compute mbtypes and the representation
        # Reinitialize descriptor
        for key in self.descr_train.keys():
            self.descr_train[key] = []
        # Check that mbtypes exists
        if self.mbtypes is None:
            raise ValueError("mbtypes missing")
        for i,mol in enumerate(self.qml_mols):
            mol.generate_slatm(self.mbtypes, local=True)
            for j,at in enumerate(self.qml_filter_ele[i]):
                if at == 1:
                    # Do include the descriptor
                    self.descr_train[mol.atomtypes[j]].append(mol.representation[j])
        for e in  self.descr_train.keys():
            size_training = len(self.target_train[e])
            # self.normalize(e)
            if len(self.descr_train[e]) > 0:
                logger.info("Training set size: %d atoms; %d molecules" % (size_training,
                    self.num_mols_train[e]))
                #print("Training set size: %d atoms; %d molecules" % (size_training,
                #    self.num_mols_train[e]))
                tgt_prop = [[self.target_train[e][i][mtp_rank][coeff]
                            for mtp_rank in range(3)
                            for coeff in range(self.max_coeffs[mtp_rank])]
                            for i in range(len(self.target_train[e]))]
                pairwise_dists = squareform(pdist(self.descr_train[e],
                    constants.ml_metric[self.kernel]))
                logger.info("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
                    % (size_training, size_training, 8*size_training**2/1e9))
                #print("building kernel matrix of size (%d,%d); %7.4f Gbytes" \
                #    % (size_training, size_training, 8*size_training**2/1e9))
                power  = constants.ml_power[self.kernel]
                prefac = constants.ml_prefactor[self.kernel]
                kmat = scipy.exp(- pairwise_dists**power / (prefac*self.krr_sigma**power))
                #kmat += self.krr_lambda*np.identity(len(self.target_train[e]))
                kmat.flat[::len(self.target_train[e])+1] += self.krr_lambda 

                self.alpha_train[e] = np.linalg.solve(kmat,tgt_prop)
        logger.info("training of multipoles finished.")
        return None

    def predict_mol(self, _system, charge=0, xyz=None):
        '''Predict multipoles in local reference frame given descriptors.'''
        tp = time.time()
        _system.initialize_multipoles()
        _system.compute_basis()

        if self.ref_mtp:
            xyz = _system.xyz[0].split('/')[-1].strip('.xyz')
            tail = '.' + xyz.split('.')[-1]
            xyz = xyz.replace('gold.','', 1)
            reffile = self.ref_path + xyz + '-mtp.txt'
            extract_file = utils.read_file(reffile)
         #   _system.multipoles = [np.array([
         #                   float(extract_file[i].split()[0]),
         #                   float(extract_file[i].split()[1]),
         #                   float(extract_file[i].split()[2]),
         #                   float(extract_file[i].split()[3]),
         #                   float(extract_file[i].split()[4]),
         #                   float(extract_file[i].split()[5]),
         #                   float(extract_file[i].split()[6]),
         #                   float(extract_file[i].split()[7]),
         #                   float(extract_file[i].split()[8])])
         #                       for i in range(len(extract_file))]
            _system.multipoles = [np.array([
                            float(extract_file[i].split()[4]),
                            float(extract_file[i].split()[6]),
                            float(extract_file[i].split()[7]),
                            float(extract_file[i].split()[5]),
                            float(extract_file[i].split()[8]),
                            float(extract_file[i].split()[9]),
                            float(extract_file[i].split()[10]),
                            float(extract_file[i].split()[11]),
                            float(extract_file[i].split()[12])])
                                for i in range(4,len(extract_file))]
        else:
            _system.build_slatm(self.mbtypes, xyz=xyz)
            power  = constants.ml_power[self.kernel]
            prefac = constants.ml_prefactor[self.kernel]
            for e in self.alpha_train.keys():
                if self.alpha_train[e] is not None:
                    pairwise_dists = cdist(_system.slatm, \
                        self.descr_train[e], constants.ml_metric[self.kernel])
                    kmat = scipy.exp(- pairwise_dists**power / (prefac*self.krr_sigma**power))
                    pred = np.dot(kmat,self.alpha_train[e])
                    for i in range(_system.num_atoms):
                        if _system.elements[i] == e:
                            _system.mtp_expansion[i] = pred[i]
            # Revert normalization
            # self.rev_normalize(_system)
            # Correct to get integer charge
            if self.correct_charge:
                # Weigh by ML error
                mol_mu = sum([constants.ml_chg_correct_error[ele]
                                for ele in _system.elements])
                totalcharge = sum([mtp[0] for mtp in _system.mtp_expansion])
                excess_chg = totalcharge - float(charge)
                if mol_mu > 0.:
                    for i,mtp_i in enumerate(_system.mtp_expansion):
                        w_i = constants.ml_chg_correct_error[_system.elements[i]]
                        mtp_i[0] += -1.*excess_chg * (w_i/mol_mu)
            # Compute multipoles from basis set expansion
            _system.expand_multipoles()
        logger.debug("Predicted multipole expansion for %s" % ( _system.xyz[0]))

        print("    Time spent predicting multipoles: %8.3f s" % (time.time() - tp))
        return None

    def add_mol_to_training(self, new_system, pun, atom=None, xyz=None):
        'Add molecule to training set'
        new_system.initialize_multipoles()

        # Don't build SLATM yet, only add information to mbtypes
        mol = None
        if new_system.xyz[0] is None:
            if xyz is not None:
                mol = qml.Compound(xyz)
            else:
                raise ValueError("Missing xyz file")
        else:
            mol = qml.Compound(new_system.xyz[0])
        self.qml_mols.append(mol)
        if atom is None:
            self.qml_filter_ele.append([1 for i in range(mol.natoms)])
        else:
            self.qml_filter_ele.append([1 if (str(mol.atomtypes[i]) == atom) else 0
                                for i in range(mol.natoms)])
        new_system.multipoles = np.empty((new_system.num_atoms,9))
        # Read in multipole moments from txt file
        new_system.load_mtp_from_hipart(pun, rotate=False)
        if len(new_system.multipoles) != new_system.num_atoms:
            raise Exception("Wrong number of charges in %s" % (pun))

        for i in range(len(new_system.elements)):
            ele_i = new_system.elements[i]
            if (ele_i == atom) or atom is None:
                #print(ele_i)
                if ele_i not in self.target_train.keys():
                    self.target_train[ele_i] = []
                    self.descr_train[ele_i] = []
                    self.num_mols_train[ele_i] = 0
                new_target_train = []
                # Rotate system until atom pairs point in all x,y,z directions
                vec_all_dir = new_system.compute_basis()
                # charge
                new_target_train.append([new_system.multipoles[i][0]])
                # dipole
                new_target_train.append(np.dot(new_system.multipoles[i][1:4],
                                    new_system.basis[i].T))
                # quadrupole
                tmp = np.dot(np.dot(new_system.basis[i],
                    utils.spher_to_cart(new_system.multipoles[i][4:9])),
                            new_system.basis[i].T).reshape((9,))
                new_target_train.append(tmp)
                self.target_train[ele_i].append(new_target_train)
            if atom in new_system.elements or atom is None:
                self.num_mols_train[ele_i] += 1
        logger.info("Added file to training set: %s" % new_system)
        return None

