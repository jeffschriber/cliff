#!/usr/bin/env python
#
# Polarizability class. Compute atomic and molecular polarizabilities.
#
# Tristan Bereau (2017)

import numpy as np
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
import logging
import math
import cliff.helpers.constants

# Set logger
logger = logging.getLogger(__name__)


class Polarizability:
    'Polarizability class. Compute atomic and molecular polarizabilities.'

    def __init__(self, sys):
        self.system = sys
        logger.setLevel(self.system.get_logger_level())
        self.num_atoms = self.system.num_atoms
        self.elements = self.system.elements
        self.csix_coeff = None
        self.freq_free_atom = None
        self.freq_scaled = None
        self.freq_scaled_vec = None
        self.pol_scaled = None
        self.pol_scaled_vec = None
        self.pol_mol_iso = None
        self.pol_mol_vec = None
        self.pol_mol_fracaniso = None

    def compute_csix(self):
        'Compute C6 coefficients'
        hirshfeld_ratios = self.system.hirshfeld_ratios
        if hirshfeld_ratios is None:
            logger.error("Assign Hirshfeld ratios first")
        self.csix_coeff = np.zeros(self.num_atoms)
        self.freq_free_atom = np.zeros(self.num_atoms)
        self.freq_scaled = np.zeros(self.num_atoms)
        self.freq_scaled_vec = np.empty((self.num_atoms,3))
        self.pol_scaled = np.zeros(self.num_atoms)
        # Free atom frequencies
        for i,ele in enumerate(self.elements):
            self.freq_free_atom[i] = 4/3. * constants.csix_free[ele] \
                / constants.pol_free[ele]**2
            # exponent 4/3 according to AT
            self.pol_scaled[i] = hirshfeld_ratios[i]  **(4/3.) \
                * constants.pol_free[ele]
            self.freq_scaled[i] = 4/3. * constants.csix_free[ele] \
                / self.pol_scaled[i]**2
            self.freq_scaled_vec[i] = np.array([self.freq_scaled[i],
                self.freq_scaled[i], self.freq_scaled[i]])
        self.csix_coeff = 3/4. * np.multiply(self.freq_free_atom,
            np.multiply(self.pol_scaled, self.pol_scaled))
        logger.info("Scaled isotropic polarizability:   %s" % self.pol_scaled)
        logger.info("Characteristic frequencies free:   %s" % self.freq_free_atom)
        logger.info("Characteristic frequencies scaled: %s" % self.freq_scaled)
        logger.info("C_six:                             %s" % self.csix_coeff)
        return None

    def compute_freq_scaled_anisotropic(self):
        'Compute anisotropic characteristic frequencies'
        if self.pol_scaled is None:
            self.compute_csix()
        self.pol_scaled_vec = np.zeros((self.num_atoms,3))
        exponent = self.system.get_pol_exponent()
        for i in range(self.num_atoms):
            rvdw_i = constants.rad_free[self.system.elements[i]]*constants.b2a
            anst_fac = np.array([0.,0.,0.])
            anst_num = np.array([0, 0, 0])
            # for j in range(1,min(3,len(self.system.atom_reorder[i]))):
            for j in range(1,len(self.system.atom_reorder[i])):
                atj = self.system.atom_reorder[i][j]
                rvdw_j = constants.rad_free[self.system.elements[atj]]*constants.b2a
                vecij = self.system.coords[atj]-self.system.coords[i]
                # Sum of vdw radii between i and j
                rvdw = rvdw_i + rvdw_j
                k_fac = np.linalg.norm(vecij)/rvdw
                anst_fac += exponent*(np.linalg.norm(vecij)/rvdw)*vecij/np.linalg.norm(vecij)
                anst_num += 1
            for k in range(3):
                if anst_num[k] > 0:
                    anst_fac[k] = 1-abs(anst_fac[k])/anst_num[k]
            mean = np.mean(anst_fac)
            for k in range(3):
                anst_fac[k] += 1-mean
            self.pol_scaled_vec[i] = self.pol_scaled[i]*anst_fac
            self.freq_scaled_vec[i] = self.freq_scaled[i] * \
                self.pol_scaled[i]**2 / \
                np.multiply(self.pol_scaled_vec[i],
                            self.pol_scaled_vec[i])
            logger.info(
                "atomic polarizability tensor of atom %d: %7.4f %7.4f %7.4f" %
                (i,self.pol_scaled_vec[i][0],
                    self.pol_scaled_vec[i][1],
                    self.pol_scaled_vec[i][2]))
        return None

    def short_range_coupling(self):
        """
        Compute short range SCS
        """
        if self.pol_scaled is None:
            self.compute_csix()
        size = (3*self.num_atoms,3*self.num_atoms)
        int_mat = np.zeros(size)
        c_mat = np.zeros(size)
        radius = self.system.get_mbd_radius()
        beta = self.system.get_mbd_beta()
        scs_cutoff = self.system.Config.get("polarizability","scs_cutoff")
        for ati in range(self.num_atoms):
            for atj in range(self.num_atoms):
                if ati is atj:
                    for ri in range(3):
                        int_mat[3*ati+ri,3*ati+ri] = self.freq_scaled[ati]**2
                        c_mat[3*ati+ri,3*ati+ri]   = 1./(
                            self.pol_scaled[ati] * \
                            self.freq_scaled_vec[ati][ri]**2)
                else:
                    for ri in range(3):
                        for rj in range(3):
                            rij = (self.system.coords[ati] - \
                                self.system.coords[atj])*constants.a2b
                            rijn = np.linalg.norm(rij)
                            # Kronecker delta between two coordinates
                            delta_ab = 1.0 if ri == rj else 0.0
                            # Compute effective width sigma
                            sigma = radius * (self.radius_vdw(ati) + \
                                self.radius_vdw(atj))
                            frac = (rijn/sigma)**beta
                            expf = math.exp(-frac)
                            int_mat[3*ati+ri,3*atj+rj] = \
                                self.freq_scaled[ati] * \
                                self.freq_scaled[atj] * \
                                math.sqrt(self.pol_scaled[ati] * \
                                    self.pol_scaled[atj]) * (
                                    (-3.*rij[ri]*rij[rj] +rijn**2*delta_ab) \
                                    /rijn**5 * (1.-cutoff(rijn, sigma, scs_cutoff)) \
                                    * (1 - expf - beta*frac*expf) + \
                                    (beta*frac+1-beta)*beta*frac* \
                                    rij[ri]*rij[rj]/rijn**5*expf )
        # Compute eigenvalues
        eigvals,eigvecs = np.linalg.eigh(int_mat)
        for i in range(3*self.num_atoms):
            eigvecs[:,i] /= math.sqrt( np.dot( np.dot(
                eigvecs.transpose()[i],c_mat),
                eigvecs.transpose()[i]))
        # Group eigenvectors into components
        aggr = sum(eigvecs[:,i]*eigvecs[:,i]/eigvals[i] for i in \
            range(len(eigvecs)))
        amol = np.zeros(3)
        for i,ele in enumerate(self.elements):
            self.pol_scaled[i] = sum(aggr[3*i:3*i+2])/3.
            self.freq_scaled[i] = 4/3. * constants.csix_free[ele] \
                / self.pol_scaled[i]**2
            for j in range(3):
                amol[j] += aggr[3*i+j]
        # print aggr.reshape((len(self.elements),3))
        # print np.array([sum(aggr.reshape((self.num_atoms,3))[i][j] for j in range(3))/3. for i in range(self.num_atoms)])
        logger.info("isotropic atomic polarizabilities: %s" % self.pol_scaled)
        # Molecular polarizability
        self.pol_mol_iso = sum(amol)/3.
        logger.info("isotropic molecular polarizability: %7.4f" % \
            self.pol_mol_iso)
        self.pol_mol_vec = np.array([amol[0],amol[1],amol[2]])
        logger.info("molecular polarizability tensor: %7.4f %7.4f %7.4f" % \
            (self.pol_mol_vec[0],self.pol_mol_vec[1],self.pol_mol_vec[2]))
        # Fractional anisotropy
        self.pol_mol_fracaniso = math.sqrt(0.5 * ((amol[0]-amol[1])**2 + \
            (amol[0]-amol[2])**2 + (amol[1]-amol[2])**2) \
            / (amol[0]**2 + amol[1]**2 + amol[2]**2))
        logger.info("Fractional anisotropy: %7.4f" % self.pol_mol_fracaniso)
        return None


    def get_csix_coeff(self):
        if self.csix_coeff is None:
            self.compute_csix()
        return self.csix_coeff

    def get_pol_scaled(self):
        if self.pol_scaled is None:
            self.compute_csix()
        return self.pol_scaled

    def get_pol_scaled_vec(self):
        if self.pol_scaled_vec is None:
            self.compute_freq_scaled_anisotropic()
        return self.pol_scaled_vec

    def radius_vdw(self, atom_id):
        'vdW radius of atom atom_id'
        atom_type = self.system.elements[atom_id]
        if self.pol_scaled_vec is None:
            self.compute_freq_scaled_anisotropic()
        return (self.pol_scaled[atom_id] / \
            constants.pol_free[atom_type])**(1/4.) \
            * constants.rad_free[atom_type]

def cutoff(rij, sigma, scs_cutoff):
    """
    cutoff for range-separated SCS
    """
    return float(1./(1+math.exp(-float(scs_cutoff)*((rij/sigma)-1.))))
