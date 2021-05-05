#!/usr/bin/env python

import qcelemental as qcel
import numpy as np
import cliff.helpers.utils as utils
import logging
import cliff.helpers.constants as constants
import os
import copy
import re
import qml
import configparser


class System:
    'Common system class for molecular system'

    def __init__(self, options, xyz=None, log=False):
        
        self.master_elements   = ['C','N','O','H','S','Cl','F','Br']
        self.master_atom_types = ['Cl','F','S1','S2','HS','HC','HN','HO','C4','C3','C2','N3','N2','N1','O1','O2','Br']

        self.name = ""

        self.xyz = [xyz]
        # Coordinates
        self.coords = None
        # Number of atoms in the molecule
        self.num_atoms = 0
        # Chemical elements
        self.elements = None
        self.Z = None
        self.atom_reorder = None
        # slatm
        self.slatm = None
        # Predict ratios
        self.hirshfeld_ratios = None
        # Atomic valence widths
        self.valence_widths = None
        # Core charge
        self.chg_core = None
        # multipoles
        self.multipoles = []
        # Expansion coefficients of multipoles along pairwise vectors
        self.mtp_expansion = None
        # Basis set for expansion
        self.basis = []
        # Principal axes for MTPs
        self.principal_axes = []
        # Pairwise vectors and rotation matrices for atomic environment descriptor
        self.pairwise_vec = []
        self.pairwise_norm = []
        self.rot_mat = []
        # Atom types
        self.atom_types = None
        # List of bonds to each atom
        self.bonded_atoms = None
        if xyz is not None:
            self.load_xyz(options.logger)
        self.atom_reorder = []
    
        self.mtp_to_disk = options.multipole_save_to_disk
        self.mtp_save_path = options.multipole_save_path

    def set_mtp_save_path(self, path):
        self.mtp_save_path = path

    def __add__(self, sys):
        """Combine two systems"""
        s = copy.deepcopy(self)
        s.elements = self.elements + sys.elements
        s.num_atoms = self.num_atoms + sys.num_atoms
        s.coords = np.append(s.coords,sys.coords).reshape((s.num_atoms,3))
        s.atom_types = self.atom_types + sys.atom_types
        s.xyz = [self.xyz[0],sys.xyz[0]]
        return s

    def __str__(self):
        if self.name != "":
            return self.name
        else:
            raise Exception("No molecule name!")

    def load_xyz(self, logger):
        extract_file = utils.read_file(self.xyz[0])
        self.num_atoms = int(extract_file[0])
        self.elements = [str(line.split()[0])
                        for i,line in enumerate(extract_file)
                        if i>1 and i<self.num_atoms+2]

        # check element types:
        for ele in self.elements:
            if ele not in self.master_elements: 
                logger.info("    Found element %s, but not parameterized!" % ele)
                raise Exception("    Found element %s, but not parameterized!" % ele)

        for i in range(len(self.elements)):
            ele = self.elements[i]
            if ele == "CL":
                self.elements[i] = "Cl"
            if ele == "BR":
                self.elements[i] = "Br"
        iterable = (float(line.split()[j])
                        for i,line in enumerate(extract_file)
                        for j in range(1,4)
                        if i>1 and i<self.num_atoms+2)
        self.coords = np.fromiter(iterable,
                    np.float).reshape(self.num_atoms, 3)
        iterable = (float(line.split()[4])
                        for i,line in enumerate(extract_file)
                        if i>1 and len(line.split())>4)
        self.identify_atom_types(logger)
        logger.debug('Loaded molecule %s with %s atoms.' \
            % (self.xyz, self.num_atoms))
        logger.debug('Elements %s' % ', '.join(self.elements))
        return None

    def load_qcel_mol(self,mol,name):
        self.name = name 
        self.num_atoms = len(mol.symbols)
        self.coords = mol.geometry*qcel.constants.conversion_factor("bohr", "angstrom")
        self.elements = mol.symbols
        self.Z = np.array([constants.atomic_number[ele] for ele in self.elements]) 
        self.identify_atom_types() 

    def build_coulomb_matrices(self, max_neighbors, atom, direction=None):
        self.coulomb_mat = []
        self.atom_reorder = []
        #for at in range(len(self.elements)):
            # Only do one atom at a time
        #if self.elements[at] == atom or (atom is None):
        coul_mat, reorder_atoms = utils.build_coulomb_matrix(
            self.coords, self.elements, atom, max_neighbors, direction)
        self.coulomb_mat.append(coul_mat)
        self.atom_reorder.append(reorder_atoms)
        return None

    def build_slatm(self, mbtypes, cutoff, xyz=None):
        #self.slatm = []
        ## Need xyz
        #if len(self.xyz) == 0:
        #    if xyz is not None:
        #        mol = qml.Compound(xyz)
        #    else:
        #        raise ValueError("Missing xyz file")
        #else:
        #    mol = qml.Compound(self.xyz[0])
        #mol.generate_slatm(mbtypes, rcut=cutoff,local=True)
        #self.slatm = mol.representation

        self.slatm = qml.representations.generate_slatm(self.coords,self.Z,mbtypes,rcut=cutoff,local=True)
        
        return None

    def initialize_multipoles(self):
        '''Initialize multipoles to 0'''
        self.multipoles = np.zeros((self.num_atoms,9))
        # mtp_expansion has size 1+3+9=13 for ranks 0,1,2.
        self.mtp_expansion = np.zeros((self.num_atoms,13))
        # MTPs based on derivatives. 1+3+6
        self.multipoles_grads = np.zeros((self.num_atoms,10))
        return None

    def expand_multipoles(self):
        """
        Expand coefficients along basis set to compute multipoles
        """
        self.multipoles = np.zeros((self.num_atoms,9))
        for i in range(self.num_atoms):
            self.multipoles[i][0] = self.mtp_expansion[i][0]
            # dipole
            if np.linalg.norm(self.mtp_expansion[i][1:4]) > 0.:
                dip = np.dot(self.mtp_expansion[i][1:4], self.basis[i])
                for j in range(3):
                    self.multipoles[i][1+j] = dip[j]
            # quadrupole
            if np.linalg.norm(self.mtp_expansion[i][4:]) > 0.:
                quadloc = self.mtp_expansion[i][4:].reshape((3,3))
                quad = utils.cart_to_sphere(np.dot(np.dot(
                        self.basis[i].T,
                    self.mtp_expansion[i][4:].reshape((3,3))),self.basis[i]))
                # quadloc = utils.spher_to_cart(self.mtp_expansion[i][4:])
                # quad = utils.cart_to_spher(np.dot(np.dot(
                #         np.linalg.inv(self.basis[i]), quadloc),
                #         self.basis[i]), stone_convention=True)
                for j in range(5):
                    self.multipoles[i][4+j] = quad[j]
        # print mtps to ref files
        if self.mtp_to_disk:
            self.save_mtp()

    def save_mtp(self):
        xyz = self.xyz[0].split('/')[-1].strip('.xyz')
        reffile = self.mtp_save_path + xyz + '-mtp.txt'

        with open(reffile, 'w') as mtp_file:
            for mtps in self.multipoles:
                for mtp in mtps:
                    mtp_file.write(str(mtp) + "\t")
                mtp_file.write("\n")

        return None

    def compute_basis(self):
        """
        Basis for multipole expansion
        """
        self.basis = []
        vec_all_dir = []
        for i in range(self.num_atoms):
            bas, vec = utils.neighboring_vectors(self.coords,
                self.elements, i)
            self.basis.append(bas)
            vec_all_dir.append(vec)
        return vec_all_dir

    def identify_atom_types(self, logger=None):
        "Identifies the atom type and bonds of every atom in the molecule"
        self.atom_types = []
        self.bonded_atoms = []
        for at_id in range(self.num_atoms):
            at_ele = self.elements[at_id]
            at_crd = np.asarray(self.coords[at_id])

            at_type = ""
            bonded = []
            for i,at in enumerate(self.coords):
                at_i = self.elements[i]
                i_crd = np.asarray(self.coords[i])
                thrsld = 2.0

                if at_ele == 'H' or at_i == 'H':
                    thrsld = 1.5
            
                if at_ele == 'S' or at_i == 'S':
                    thrsld = 2.2

                dist = np.linalg.norm(np.subtract(at_crd,i_crd))
                if dist < thrsld and i != at_id:
                    bonded.append((at_i,at,dist))
            self.bonded_atoms.append(bonded)
            if at_ele == 'H':
                at_type = 'H'+bonded[0][0]
            elif at_ele == 'O':
                at_type = 'O'+str(len(bonded))
            elif at_ele == 'N':
                at_type = 'N'+str(len(bonded))
            elif at_ele == 'C':
                at_type = 'C'+str(len(bonded))
            elif at_ele == 'S':
                if len(bonded) >= 2:
                    at_type = 'S2'
                else:
                    at_type = 'S'+str(len(bonded))
            elif at_ele == 'Cl' or at_ele == 'CL':
                at_type = 'Cl'
            elif at_ele == 'F':
                at_type = 'F'
            elif at_ele == 'BR' or at_ele == 'Br':
                at_type = 'Br'

            if at_type not in self.master_atom_types and logger != None:
                logger.info("    Atom type %s detected, but is not parameterized!" % at_type) 
                raise Exception("    Atom type %s detected, but is not parameterized!" % at_type) 

            self.atom_types.append(at_type)
        return None

    def load_mtp_from_hipart(self, txt, rotate=False):
        """Load multipoles from hipart output text file"""
        extract_file = utils.read_file(txt)
        self.multipoles = [np.array([
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
        if rotate is True:
            self.compute_principal_axes()
            self.multipoles = utils.rotate_mtps_back(
                    self.multipoles, self.principal_axes)
        return None
