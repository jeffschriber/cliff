#!/usr/bin/env python
#
# System class. Define overall molecular system, load coordinates,
# all variables and physical parameters.
#
# Tristan Bereau (2017)

import numpy as np
import utils
import logging
from calculator import Calculator
import constants
import os
import copy
import re
import qml
import configparser

# Set logger
logger = logging.getLogger(__name__)

class System(Calculator):
    'Common system class for molecular system'

    def __init__(self, xyz=None, mps=None, log=False):
        Calculator.__init__(self, config_file = "config.ini")
        logger.setLevel(self.get_logger_level())
        # xyz and mps can't both be empty
        if not (xyz or mps):
            logger.error("Need either one xyz or mps file")
            print("Need either one xyz or mps file")
            exit(1)
        self.xyz = [xyz]
        self.mps = mps
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
        # molecular dipole found in the MPS file
        self.mps_dipole = None
        # Atom types
        self.atom_types = None
        # List of bonds to each atom
        self.bonded_atoms = None
        if len(self.xyz) == 1:
            self.load_xyz()
        elif mps:
            self.load_mps()

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
        elif self.mps:
            return self.mps
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
        self.logger.debug('Loaded molecule %s with %s atoms.' \
            % (self.xyz, self.num_atoms))
        self.logger.debug('Elements %s' % ', '.join(self.elements))
        return None

    def load_mps(self):
        """Load VOTCA-type MPS file. Only supports Rank 0."""
        extract_file = utils.read_file(self.mps)
        # Molecular dipole
        mps_dipole_tmp = extract_file[1].rstrip().split('=')[3].split()
        self.mps_dipole = np.array([float(ele)
                    for _,ele in enumerate(mps_dipole_tmp)])
        # Element and number of atoms
        self.elements = [str(line.split()[0])
                            for _,line in enumerate(extract_file)
                            if "Rank 0" in line]
        self.num_atoms = len(self.elements)
        # Coordinates
        iterable = (float(line.split()[j])
                            for _,line in enumerate(extract_file)
                            for j in range(1,4)
                            if "Rank 0" in line)
        self.coords = np.fromiter(iterable,
                np.float).reshape(self.num_atoms, 3)
        self.identify_atom_types()
        # Charges
        self.multipoles = [np.array([float(extract_file[i+1].split()[0]),
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                    for i,line in enumerate(extract_file)
                    if "Rank 0" in line]
        logger.debug("Loaded molecule %s with %s atoms" % (self.mps,
                self.num_atoms))
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

    def build_bag_of_bonds(self, bob_struct, max_neighbors):
        self.bag_of_bonds = []
        for at in range(len(self.elements)):
            bob, reorder = utils.build_bag_of_bonds(self.coords, self.elements, \
                at, bob_struct, max_neighbors)
            self.bag_of_bonds.append(bob)
            if len(self.atom_reorder) < self.num_atoms:
                self.atom_reorder.append(reorder)
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

    def compute_voronoi(self):
        '''Estimates MTP coefficients from Voronoi for atom atom (ID) on a discrete
        set of points.'''
        self.voronoi_baseline = []
        if self.Config.get("multipoles","voronoi") in \
            ["on","yes","True","true"]:
            a2b = constants.a2b
            grid_max = self.Config.getfloat("multipoles","voronoi_grid_max")
            grid_step = self.Config.getfloat("multipoles","voronoi_grid_step")
            # Work in bohr
            b_coords = self.coords*a2b
            for i in range(len(self.elements)):
                # Grid-point boundaries
                xmin = b_coords[i][0] - grid_max
                ymin = b_coords[i][1] - grid_max
                zmin = b_coords[i][2] - grid_max
                xmax = xmin + 2*grid_max
                ymax = ymin + 2*grid_max
                zmax = zmin + 2*grid_max
                # Initialize vector of coefficients
                v_coeffs = np.zeros(10)
                x0 = xmin
                weightar = []
                while x0 < xmax:
                    y0 = ymin
                    while y0 < ymax:
                        z0 = zmin
                        while z0 < zmax:
                            pos  = np.array([x0,y0,z0])
                            rvec = np.array(pos-b_coords[i])
                            n_a_free = np.ones(10)* \
                                utils.atom_dens_free(b_coords[i], \
                                    self.elements[i], pos, i)
                            dist_term = np.array([1.,rvec[0],rvec[1],rvec[2],
                                rvec[0]*rvec[0],rvec[0]*rvec[1],rvec[0]*rvec[2],
                                rvec[1]*rvec[1],rvec[1]*rvec[2],rvec[2]*rvec[2]])
                            fac = np.multiply(dist_term, n_a_free)
                            # Voronoi
                            closest_atm = i
                            shortest_dis = 1000.0
                            for atomj in range(self.num_atoms):
                                atom_dis = np.linalg.norm(pos-b_coords[atomj])
                                if atom_dis < shortest_dis:
                                    closest_atm = atomj
                                    shortest_dis = atom_dis
                            if closest_atm == i:
                                v_coeffs += fac
                            # Update coordinates
                            z0 += grid_step
                        y0 += grid_step
                    x0 += grid_step
                # Convert to spherical coordinates
                quad_cart = np.array([v_coeffs[4],v_coeffs[5],v_coeffs[6],
                   0.,v_coeffs[7],v_coeffs[8],0.,0.,v_coeffs[9]]).reshape(3,3)
                quad_sph  = utils.cart_to_spher(quad_cart)
                self.voronoi_baseline.append([v_coeffs[0], \
                    np.array([v_coeffs[1],v_coeffs[2],v_coeffs[3]]), \
                    quad_sph])
                logger.debug("Voronoi coeffs for atom %s (ID: %d):\n %s" % \
                    (self.elements[i],i, \
                    np.hstack([v_coeffs[0], \
                    np.array([v_coeffs[1],v_coeffs[2],v_coeffs[3]]), \
                    quad_sph])))
        else:
             for i in range(len(self.elements)):
                 self.voronoi_baseline.append([0., np.zeros(3), np.zeros(5)])
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

    def load_mtp_from_poltype(self, txt):
        """Load global multipoles from poltype output file"""
        extract_file = utils.read_file(txt)
        i_line = [i for i in range(len(extract_file)) if "Global" in extract_file[i]]
        if len(i_line) == 0:
            print("Can't find Global multipoles in ",txt)
            exit(1)
        i_line = i_line[0]
        mtps = []
        mtp_i = np.zeros(9)
        while "Local" not in extract_file[i_line]:
            line = extract_file[i_line].split()
            if len(line) > 1:
                if line[0] == 'Charge:':
                    mtp_i[0] = float(line[1])
                if line[0] == 'Dipole:':
                    for j,k in zip([1,2,3],[1,2,3]):
                        mtp_i[j] = float(line[k])
                if line[0] == 'Quadrupole:':
                    quad = np.zeros((3,3))
                    for j1,j2,l,k in zip([0,1,1,2,2,2],[0,0,1,0,1,2],[0,1,1,2,2,2],[1,0,1,0,1,2]):
                        quad[j1][j2] = float(extract_file[i_line+l].split()[k])
                    quad = utils.symmetrize(quad)
                    mtp_i[4:9] = utils.cart_to_spher(quad, stone_convention=True)
                    mtps.append(mtp_i)
                    mtp_i = np.zeros(9)
            i_line += 1
        self.multipoles = mtps
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
        if self.Config.get('multipoles','save_to_disk'):
            refpath = self.Config.get('multipoles','mtp_save_path') 
            xyz = self.xyz[0].split('/')[-1].strip('.xyz')
            reffile = refpath + xyz + '-mtp.txt'
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
