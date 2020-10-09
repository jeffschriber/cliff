#!/usr/bin/env python
#
# Polarizability class. Compute atomic and molecular polarizabilities.
#
import numpy as np
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
import logging
import math
import cliff.helpers.constants as constants



class Polarizability:
    'Polarizability class. Compute atomic and molecular polarizabilities.'

    def __init__(self, name, logger, scs_cutoff, pol_exponent, sys):

        # Set logger
        self.logger = logger

        self.system = sys
        self.num_atoms = self.system.num_atoms
        self.elements = self.system.elements

        self.freq_free_atom = None
        self.freq_scaled = None
        self.freq_scaled_vec = None
        self.pol_scaled = None
        self.pol_scaled_vec = None
        self.pol_mol_iso = None
        self.pol_mol_vec = None
        self.pol_mol_fracaniso = None
        self.scs_cutoff = scs_cutoff
        self.exponent = pol_exponent


    def compute_freq_pol(self):
        'Compute characteristic frequency and static polarizabilities'
        hirshfeld_ratios = self.system.hirshfeld_ratios
        if hirshfeld_ratios is None:
            self.logger.error("Assign Hirshfeld ratios first")
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

        self.logger.debug("Scaled isotropic polarizability:   %s" % self.pol_scaled)
        self.logger.debug("Characteristic frequencies free:   %s" % self.freq_free_atom)
        self.logger.debug("Characteristic frequencies scaled: %s" % self.freq_scaled)
        return None

    def compute_freq_scaled_anisotropic(self):
        'Compute anisotropic characteristic frequencies'
        if self.pol_scaled is None:
            self.compute_freq_pol()
        self.pol_scaled_vec = np.zeros((self.num_atoms,3))
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
                anst_fac += self.exponent*(np.linalg.norm(vecij)/rvdw)*vecij/np.linalg.norm(vecij)
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
            self.logger.debug(
                "atomic polarizability tensor of atom %d: %7.4f %7.4f %7.4f" %
                (i,self.pol_scaled_vec[i][0],
                    self.pol_scaled_vec[i][1],
                    self.pol_scaled_vec[i][2]))
        return None

    def get_pol_scaled(self):
        if self.pol_scaled is None:
            self.compute_freq_pol()
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
