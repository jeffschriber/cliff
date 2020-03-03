#!/usr/bin/env python
#
# InductionCalc class. Compute induction.
#
# Tristan Bereau (2017)

import numpy as np
from cliff.helpers.system import System
from cliff.components.cp_multipoles import CPMultipoleCalc, interaction_tensor
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.polarizability import Polarizability
from cliff.helpers.cell import Cell
from numpy import exp
from copy import deepcopy
import logging
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
from numba import jit

# Set logger
logger = logging.getLogger(__name__)

class InductionCalc(CPMultipoleCalc):

    def __init__(self, options, sys, cell, ind_sr=None, hirshfeld_pred="krr", v1=False):
        CPMultipoleCalc.__init__(self,options,sys,cell)
        logger.setLevel(options.logger_level)
        self.cell = cell
        self.induced_dip = None
        self.energy_polarization = 0.0
        self.energy_shortranged = 0.0
        self.sys_comb = sys

        self.omega = options.indu_omega
        self.conv  = options.indu_conv        
        self.ind_sr = options.indu_sr_params
        self.smearing_coeff = options.indu_smearing_coeff
        self.scs_cutoff = options.pol_scs_cutoff
        self.pol_exponent = options.pol_exponent

    def add_system(self, sys):
        CPMultipoleCalc.add_system(self, sys)
        self.sys_comb.hirshfeld_ratios = np.append(self.sys_comb.hirshfeld_ratios,
            sys.hirshfeld_ratios)
        self.sys_comb.populations, self.sys_comb.valence_widths = [], []
        # Refinement
        for s in self.systems:
            self.sys_comb.populations    = np.append(self.sys_comb.populations,
                                                        s.populations)
            self.sys_comb.valence_widths = np.append(self.sys_comb.valence_widths,
                    s.valence_widths)
        return None

    def polarization_energy(self,options, smearing_coeff=None, stone_convention=False):
        """Compute induction energy"""
        self.convert_mtps_to_cartesian(stone_convention)
        # Populate Hirshfeld ratios of combined systems
        # self.hirshfelds = ...
        # Setup list of atoms to sum over
        atom_coord = [crd for sys in self.systems
                            for _, crd in enumerate(sys.coords)]

        atom_ele   = [ele for sys in self.systems
                            for _, ele in enumerate(sys.elements)]
        atom_typ   = [typ for sys in self.systems
                            for _,typ in enumerate(sys.atom_types)]
        #print("Types", atom_typ)
        populations = [p for _,p in enumerate(self.sys_comb.populations)]
        valwidths   = [v/constants.a2b
                       for _,v in enumerate(self.sys_comb.valence_widths)]
        self.induced_dip = np.zeros((len(atom_ele),3))
        # Atomic polarizabilities
        atom_alpha_iso = [alpha for _, alpha in enumerate(
                                Polarizability(self.scs_cutoff,self.pol_exponent,self.sys_comb).get_pol_scaled())]

        # Short-range correction 
        self.energy_shortranged = 0.0 #-0.0480131753842 #comes from linear coefficient

        for i, c_i in enumerate(atom_coord):
            for j, c_j in enumerate(atom_coord):

                if (self.different_mols(i,j) and i < j):
                    #print "HERE", atom_typ[i], atom_typ[j]
                    fmbis = utils.slater_mbis(self.cell,
                    atom_coord[i], populations[i], valwidths[i], self.ind_sr[atom_typ[i]],
                    atom_coord[j], populations[j], valwidths[j], self.ind_sr[atom_typ[j]])
 
                    #self.energy_shortranged +=  self.ind_sr[atom_typ[i]] * self.ind_sr[atom_typ[j]] * fmbis #/ pair_count[pairs_key.index(pair)]
                    self.energy_shortranged += fmbis #/ pair_count[pairs_key.index(pair)]
        logger.info("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)
        # Intitial induced dipoles
        if smearing_coeff:
            self.smearing_coeff = smearing_coeff 

        for i,_ in enumerate(atom_ele):
            self.induced_dip[i] = sum([atom_alpha_iso[i] *
                        self.interaction_permanent_multipoles(atom_coord[i],
                            atom_coord[j], atom_alpha_iso[i], atom_alpha_iso[j],
                            self.smearing_coeff, self.mtps_cart[j])
                            for j,_ in enumerate(atom_ele)
                            if self.different_mols(i,j)])
        # Self-consistent polarization
        mu_next = deepcopy(self.induced_dip)
        mu_prev = np.ones((len(atom_ele),3))
        diff_init = np.linalg.norm(mu_next-mu_prev)
        counter = 0
        while np.linalg.norm(mu_next-mu_prev) > self.conv:
            mu_prev = deepcopy(mu_next)
            for i,_ in enumerate(atom_ele):
                mu_next[i] = (1-self.omega)*mu_prev[i] + self.omega * (self.induced_dip[i] + sum(
                                [atom_alpha_iso[i] *
                                    product_smeared_ind_dip(atom_coord[i],
                                        atom_coord[j], self.cell,
                                        atom_alpha_iso[i], atom_alpha_iso[j],
                                        self.smearing_coeff, mu_prev[j])
                                    for j,_ in enumerate(atom_ele) if i != j]))
            counter += 1
            if np.linalg.norm(mu_next-mu_prev) > diff_init*10 or counter > 2000:
                print("Can't converge self-consistent equations. Exiting.")
                exit(1)
            if counter % 50 == 0 and self.omega > 0.2:
                self.omega *= 0.8
        self.induced_dip = np.zeros((len(atom_ele),13))
        logger.info("Converged induced dipoles [debye]:")
        for i,_ in enumerate(atom_ele):
            logger.info("Atom %d: %7.4f %7.4f %7.4f" % (i,
                            mu_next[i][0] * constants.au2debye,
                            mu_next[i][1] * constants.au2debye,
                            mu_next[i][2] * constants.au2debye))
            self.induced_dip[i][1:4] = mu_next[i]

        self.energy_polarization = 0.5 * constants.au2kcalmol * sum([np.dot(
                    self.induced_dip[i].T,
                    np.dot(interaction_tensor(atom_coord[i], atom_coord[j], self.cell),
                        self.mtps_cart[j])) + np.dot(
                        self.mtps_cart[i].T,
                        np.dot(interaction_tensor(atom_coord[i], atom_coord[j], self.cell),
                            self.induced_dip[j]))
                            for i in range(    len(atom_ele))
                            for j in range(i+1,len(atom_ele))
                            if self.different_mols(i,j) and j>i])
        logger.info("Polarization energy: %7.4f kcal/mol" % self.energy_polarization)
        #print("Polarization energy", self.energy_polarization)
        #print "Short range", self.energy_shortranged
        #print self.energy_polarization , self.energy_shortranged
        return self.energy_polarization - self.energy_shortranged

    def different_mols(self, i, j):
        """
        Returns True if atom indices i and j belong to different systems.
        """
        return self.atom_in_system[i] is not self.atom_in_system[j]

    def interaction_permanent_multipoles(self, coord1, coord2, at_pol1, at_pol2,
        smearing, mtp_perm):
        """
        Returns product of smeared interaction tensor with permanent multipoles.
        Corresponds to the external field.
        """
        interac = np.zeros(3)
        # Permanent charge contribution + monopole charge penetration
        charge = mtp_perm[0]
        vec = self.cell.pbc_distance(coord1, coord2) * constants.a2b

        interac += [interaction_tensor_first(vec, at_pol1,
                        at_pol2, smearing, i)*charge for i in range(3)]
        # Permanent dipole contribution
        interac += [sum([interaction_tensor_second(vec, at_pol1,
                        at_pol2, smearing, i, j)*mtp_perm[1+j]
                        for j in range(3)])
                        for i in range(3)]
        # # Permanent quadrupole contribution
        interac += [sum([interaction_tensor_third(vec, at_pol1,
                        at_pol2, smearing, i, j, k)*mtp_perm[4+3*j+k]
                        for j in range(3)
                        for k in range(3)])
                        for i in range(3)]
        return interac

def product_smeared_ind_dip(coord1, coord2, cell, at_pol1, at_pol2,
    smearing, ind_dip):
    """
    Returns product of smeared interaction tensor with induced dipole.
    """
    vec = cell.pbc_distance(coord1, coord2) * constants.a2b
    return np.array([sum([interaction_tensor_second(vec, at_pol1,
                at_pol2, smearing, i, j)*ind_dip[j]
                for j in range(3)])
                for i in range(3)])

@jit
def interaction_tensor_first(vec, at_pol1, at_pol2, smearing, dir1):
    """
    Returns smeared dipole-charge interaction tensor using Thole formalism.
    """
    r = np.linalg.norm(vec)
    u = r/(at_pol1*at_pol2)**(1/6.)
    ri3 = 1./r**3
    return -(1.-np.exp(-smearing*u**3))*vec[dir1]*ri3

@jit
def interaction_tensor_second(vec, at_pol1, at_pol2, smearing,
            dir1, dir2):
    """
    Returns smeared dipole-dipole interaction tensor using Thole formalism.
    """
    r = np.linalg.norm(vec)
    u = r/(at_pol1*at_pol2)**(1/6.)
    ri  = 1./r
    ri3 = ri**3
    ri5 = ri**5
    lambda3 = 1.-np.exp(-smearing*u**3)
    lambda5 = 1.-(1+smearing*u**3) * np.exp(-smearing*u**3)
    diag = lambda3*ri3 if dir1 is dir2 else 0.
    return lambda5*3*vec[dir1]*vec[dir2]*ri5 - diag

@jit
def interaction_tensor_third(vec, at_pol1, at_pol2, smearing,
            dir1, dir2, dir3):
    """
    Returns smeared dipole-quadrupole interaction tensor using Thole formalism.
    """
    r = np.linalg.norm(vec)
    u = r/(at_pol1*at_pol2)**(1/6.)
    ri  = 1./r
    ri5 = ri**5
    ri7 = ri**7
    lambda5 = 1.-(1+smearing*u**3) * np.exp(-smearing*u**3)
    lambda7 = 1.-(1+smearing*u**3+3/5.*smearing**2*u**6) * np.exp(-smearing*u**3)
    coeff1 = vec[dir1] if dir2 is dir3 else 0.
    coeff2 = vec[dir2] if dir1 is dir3 else 0.
    coeff3 = vec[dir3] if dir1 is dir2 else 0.
    return - lambda7*15*vec[dir1]*vec[dir2]*vec[dir3]*ri7 \
            + lambda5*3*(coeff1+coeff2+coeff3)*ri5

