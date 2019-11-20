#!/usr/bin/env python
#
# System class. Define overall molecular system, load coordinates,
# all variables and physical parameters.
#
# Tristan Bereau (2017)

import numpy as np
import cliff.helpers.utils as utils
import logging
import cliff.helpers.constants
import os
import copy
import re
import qml
import configparser

# Set logger
logger = logging.getLogger(__name__)

class System:
    'Common system class for molecular system'

    def __init__(self, options, xyz=None, log=False):
        logger.setLevel(options.logger_level)
        # xyz and mps can't both be empty
        if not xyz:
            logger.error("Need an xyz file")
            exit(1)
        self.xyz = [xyz]
        # Coordinates
        self.coords = None
        # Number of atoms in the molecule
        self.num_atoms = 0
        # Chemical elements
        self.elements = None
        self.hirshfeld_ref = None
        # Coulomb matrix
        self.coulomb_mat = None
        self.atom_reorder = None
        # Coulomb matrix within environment
        self.coulmat_env = None
        # Coulomb matrix and derivatives
        self.coulmat_grads = None
        # Bag of bonds
        self.bag_of_bonds = None
        # slatm
        self.slatm = None
        # Predict ratios
        self.hirshfeld_ratios = None
        # Atomic populations
        self.populations = None
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
        # Voronoi baseline for MTPs
        self.voronoi_baseline = []
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
        if len(self.xyz) == 1:
            self.load_xyz()
    
        self.mtp_to_disk = options.multipole_save_to_disk

        if self.mtp_to_disk:
            self.mtp_save_path = options.multipole_save_path


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
        if self.xyz:
            return self.xyz
        else:
            logger.error("No molecule name!")
            print("No molecule name!")
            exit(1)

    def load_xyz(self):
        extract_file = utils.read_file(self.xyz[0])
        self.num_atoms = int(extract_file[0])
        self.elements = [str(line.split()[0])
                        for i,line in enumerate(extract_file)
                        if i>1 and i<self.num_atoms+2]
        iterable = (float(line.split()[j])
                        for i,line in enumerate(extract_file)
                        for j in range(1,4)
                        if i>1 and i<self.num_atoms+2)
        self.coords = np.fromiter(iterable,
                    np.float).reshape(self.num_atoms, 3)
        iterable = (float(line.split()[4])
                        for i,line in enumerate(extract_file)
                        if i>1 and len(line.split())>4)
        self.hirshfeld_ref = np.fromiter(iterable,np.float)
        self.identify_atom_types()
        logger.debug('Loaded molecule %s with %s atoms.' \
            % (self.xyz, self.num_atoms))
        logger.debug('Elements %s' % ', '.join(self.elements))
        return None

    def build_coulomb_matrices(self, max_neighbors, direction=None):
        self.coulomb_mat = []
        self.atom_reorder = []
        for at in range(len(self.elements)):
            coul_mat, reorder_atoms = utils.build_coulomb_matrix(
                self.coords, self.elements, at, max_neighbors, direction)
            self.coulomb_mat.append(coul_mat)
            self.atom_reorder.append(reorder_atoms)
        return None

    def build_coulmat_env(self, sys_comb, max_neighbors):
        self.coulmat_env = []
        for at in range(len(self.elements)):
            self.coulmat_env.append(utils.build_coulomb_matrix_env(self.coords, self.elements,
                at, max_neighbors, sys_comb.coords, sys_comb.elements))
        return None

    def build_coulomb_grads(self, max_neighbors):
        self.coulmat_grads = []
        self.atom_reorder = []
        for at in range(len(self.elements)):
            coulmat0, coulmat1, coulmat2, reorder_atoms = \
                utils.coulomb_with_grads(self.coords, self.elements,
                at, max_neighbors)
            self.coulmat_grads.append([coulmat0,coulmat1,coulmat2])
            self.atom_reorder.append(reorder_atoms)
        return None

    def build_slatm(self, mbtypes, xyz=None):
        self.slatm = []
        # Need xyz
        if len(self.xyz) == 0:
            if xyz is not None:
                mol = qml.Compound(xyz)
            else:
                raise ValueError("Missing xyz file")
        else:
            mol = qml.Compound(self.xyz[0])
        mol.generate_slatm(mbtypes, local=True)
        self.slatm = mol.representation
        return None

    def initialize_multipoles(self):
        '''Initialize multipoles to 0'''
        self.multipoles = np.zeros((self.num_atoms,9))
        # mtp_expansion has size 1+3+9=13 for ranks 0,1,2.
        self.mtp_expansion = np.zeros((self.num_atoms,13))
        # MTPs based on derivatives. 1+3+6
        self.multipoles_grads = np.zeros((self.num_atoms,10))
        return None

    def initialize_atomic_environment(self):
        '''Store pairwise vectors and rotation matrices for
        atomic environment descriptor'''
        self.pairwise_vec = []
        self.pairwise_norm = []
        self.rot_mat = []
        z = np.array([0.,0.,1.])
        for i in range(self.num_atoms):
            pair_i = []
            pair_r_i = []
            rot_i = []
            for j in range(self.num_atoms):
                v = self.coords[j] - self.coords[i]
                pair_i.append(v)
                pair_r_i.append(np.linalg.norm(v))
                rot_i.append(utils.ab_rotation(v, z))
            self.pairwise_vec.append(pair_i)
            self.pairwise_norm.append(pair_r_i)
            self.rot_mat.append(rot_i)
        return None

    def compute_principal_axes(self):
        '''Project MTP coefficients (except for Q00) along each atom-atom vector.
        Ordered by atom ID. Returns  principal axes.'''
        self.principal_axes = []
        mass = np.zeros(len(self.elements))
        for m in range(len(self.elements)):
            mass[m] = constants.atomic_weight[self.elements[m]]
        for i in range(len(self.coords)):
            atomi = self.coords[i]
            eigvecs = np.zeros((3,3))
            if self.num_atoms == 1:
                eigvecs = np.identity(3)
            elif self.num_atoms == 2:
                # Only one axis defined
                eigvecs[:,0] = self.coords[i]
                for j in range(len(self.coords)):
                    # Only take neighbor
                    if j==i+1:
                        eigvecs[:,0] -= self.coords[j]
                if abs(np.linalg.norm(eigvecs[:,0])) > 1e-12:
                    eigvecs[:,0] /= np.linalg.norm(eigvecs[:,0])
                # Construct other two eigenvectors
                eigvecs[:,1] = np.cross([1,1,1],eigvecs[:,0])
                if abs(np.linalg.norm(eigvecs[:,1])) > 1e-12:
                    eigvecs[:,1] /= np.linalg.norm(eigvecs[:,1])
                eigvecs[:,2] = np.cross(eigvecs[:,0],eigvecs[:,1])
                if abs(np.linalg.norm(eigvecs[:,2])) > 1e-12:
                    eigvecs[:,2] /= np.linalg.norm(eigvecs[:,2])
                # center of mass of molecule
                com = sum([mass[j]*self.coords[j] for j in \
                    range(len(self.coords))]) / sum(mass)
                for v in range(3):
                    if np.dot(com - self.coords[i],eigvecs[v]) < 0.:
                      eigvecs[v] *= -1.
            else:
                # Compute inertia tensor of molecule around atom i
                coords = self.coords - self.coords[i]
                inertia = np.dot(mass*coords.transpose(),coords)
                eigvals,eigvecs = np.linalg.eig(inertia)
                # center of mass of molecule
                com = sum([mass[j]*self.coords[j] for j in \
                    range(len(self.coords))]) / sum(mass)
            # Orient first two eigenvectors so that com is in positive quadrant.
            for v in range(3):
                if np.dot(com - self.coords[i],eigvecs[v]) < 0.:
                  eigvecs[v] *= -1.
            self.principal_axes.append(eigvecs.transpose())
            logger.debug("Principal axes for atom %s (ID: %d):\n %s" % \
                (self.elements[i],i,self.principal_axes[i]))
        return None


    def molecular_principal_components(self):
        """
        Principal components around center of mass.
        Returns sorted eigenvalues and eigenvectors.
        """
        return utils.inertia_tensor(self.coords, self.elements)

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
                quad = utils.cart_to_spher(np.dot(np.dot(
                        self.basis[i].T,
                    self.mtp_expansion[i][4:].reshape((3,3))),self.basis[i]),
                    stone_convention=True)
                # quadloc = utils.spher_to_cart(self.mtp_expansion[i][4:])
                # quad = utils.cart_to_spher(np.dot(np.dot(
                #         np.linalg.inv(self.basis[i]), quadloc),
                #         self.basis[i]), stone_convention=True)
                for j in range(5):
                    self.multipoles[i][4+j] = quad[j]
        
        # print mtps to ref files
        if self.mtp_to_disk:
            xyz = self.xyz[0].split('/')[-1].strip('.xyz')
            reffile = slef.mtp_save_path + xyz + '-mtp.txt'
            print(reffile) 
            #if os.path.exists(reffile):
            #    return None

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

    def rotate_random(self):
        """Arbitrary rotation around center of mass of coordinates
        as well as multipole moments"""
        masses = np.array([float(constants.atomic_weight[ele])
                    for _,ele in enumerate(self.elements)])
        com = np.sum([m*c for m,c in zip(masses,self.coords)],
                axis=0)/np.sum(masses)
        print("dis",np.linalg.norm(self.coords[1]-self.coords[0]),\
            np.linalg.norm(self.coords[2]-self.coords[0]))
        # Generate random rotation matrix
        rotmat = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                rotmat[i,j] = np.random.rand()
        rotmat[2] = np.cross(rotmat[0],rotmat[1])
        rotmat[0] = np.cross(rotmat[1],rotmat[2])
        for i in range(3):
            rotmat[i] /= np.linalg.norm(rotmat[i,:])
        print(self.multipoles)
        print(rotmat)
        # Update coordinates and multipoles
        for i in range(self.num_atoms):
            self.coords[i] = np.dot(rotmat, self.coords[i] - com) + com
            self.multipoles[i][1:4] = np.dot(rotmat, self.multipoles[i][1:4])
            self.multipoles[i][4:9] = utils.cart_to_spher(np.dot(np.dot(
                rotmat,utils.spher_to_cart(self.multipoles[i][4:9])), rotmat.T))
            if i == 0:
                print("rot\n",utils.spher_to_cart(self.multipoles[i][4:9]).reshape((3,3)))
                print(np.dot(np.dot(rotmat.T, utils.spher_to_cart(self.multipoles[i][4:9])),
                           rotmat))
                print(self.multipoles[i][4:9])
        return None


    def identify_atom_types(self):
        "Identifies the atom type and bonds of every atom in the molecule"
        self.atom_types = []
        self.bonded_atoms = []
        for at_id in range(self.num_atoms):
            at_ele = self.elements[at_id]
            at_crd = self.coords[at_id]
            bonded = []
            for i,at in enumerate(self.coords):
                at_i = self.elements[i]
                thrsld = 1.6 if at_ele == 'H' or at_i == 'H' else 2.0
                dist = np.linalg.norm(at_crd-self.coords[i])
                if dist < thrsld and i != at_id:
                    bonded.append((at_i,at,dist))
            self.bonded_atoms.append(bonded)
            if at_ele == 'H':
                self.atom_types.append('H'+bonded[0][0])
            elif at_ele == 'O':
                self.atom_types.append('O'+str(len(bonded)))
            elif at_ele == 'N':
                self.atom_types.append('N'+str(len(bonded)))
            elif at_ele == 'C':
                self.atom_types.append('C'+str(len(bonded)))
            elif at_ele == 'S':
                self.atom_types.append('S'+str(len(bonded)))
            elif at_ele == 'Cl' or at_ele == 'CL':
                self.atom_types.append('Cl'+str(len(bonded)))
            elif at_ele == 'F':
                self.atom_types.append('F'+str(len(bonded)))
        return None
