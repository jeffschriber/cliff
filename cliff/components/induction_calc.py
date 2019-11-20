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
from cliff.atomic_properties.atomic_density import AtomicDensity
from cliff.helpers.cell import Cell
from numpy import exp
from copy import deepcopy
import logging
import cliff.helpers.constants
import cliff.helpers.utils
from numba import jit

# Set logger
logger = logging.getLogger(__name__)

class InductionCalc:

    def __init__(self, options, sys, cell, ind_sr=None, hirshfeld_pred="krr", v1=False):
        logger.setLevel(options.get_logger_level())
        self.cell = cell
        self.induced_dip = None
        self.energy_polarization = 0.0
        self.energy_shortranged = 0.0
        # Predict Hirshfeld ratios for sys
        self.hirshfeld_pred = hirshfeld_pred
        self.hirsh = Hirshfeld(options)
        self.hirsh.load_ml()
        self.hirsh.predict_mol(sys,self.hirshfeld_pred)
        # Predict valence widths for sys
        self.adens = AtomicDensity(options)
        self.adens.load_ml()
        # self.adens.load_ml_env()
        self.adens.predict_mol(sys)
        # All molecules together
        self.sys_comb = sys
        self.adens.predict_mol(self.sys_comb)

        # Atom types for short-ranged induction
        self.ele_ad = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
        
        self.ind_sr = {}
        if ind_sr != None:
            for n,ele in enumerate(self.ele_ad):
                self.ind_sr[ele] = ind_sr[n]
        else:    
            for ele in self.ele_ad:
                self.ind_sr[ele] = self.systems[0].Config.getfloat("induction",
                                                                   "sr["+ele+"]")

    def add_system(self, sys):
        CPMultipoleCalc.add_system(self, sys)
        self.hirsh.predict_mol(sys, self.hirshfeld_pred)
        self.sys_comb.hirshfeld_ratios = np.append(self.sys_comb.hirshfeld_ratios,
            sys.hirshfeld_ratios)
        self.sys_comb.populations, self.sys_comb.valence_widths = [], []
        # Refinement
        for s in self.systems:
            # self.adens.predict_mol_env(s,self.sys_comb)
            self.sys_comb.populations    = np.append(self.sys_comb.populations,
                                                        s.populations)
            self.sys_comb.valence_widths = np.append(self.sys_comb.valence_widths,
                    s.valence_widths)
        return None

    def polarization_energy_v1(self, stone_convention=False):
        """Compute induction energy"""
        self.convert_mtps_to_cartesian(stone_convention)
        # Populate Hirshfeld ratios of combined systems
        # self.hirshfelds = ...
        omega = self.systems[0].Config.getfloat("induction","omega")
        # Setup list of atoms to sum over
        atom_coord = [crd for sys in self.systems
                            for _, crd in enumerate(sys.coords)]
        atom_ele   = [ele for sys in self.systems
                            for _, ele in enumerate(sys.elements)]
        atom_typ   = [typ for sys in self.systems
                            for _,typ in enumerate(sys.atom_types)]
        populations = [p for _,p in enumerate(self.sys_comb.populations)]
        valwidths   = [v/constants.a2b
                       for _,v in enumerate(self.sys_comb.valence_widths)]
        self.induced_dip = np.zeros((len(atom_ele),3))
        # Atomic polarizabilities
        atom_alpha_iso = [alpha for _, alpha in enumerate(
                                Polarizability(self.sys_comb).get_pol_scaled())]
        # Short-range contribution (minus sign because it's attractive)

        self.energy_shortranged = -1.*sum([self.slater_mbis(
            atom_coord[i], populations[i], valwidths[i], atom_typ[i],
            atom_coord[j], populations[j], valwidths[j], atom_typ[j])
                                       for i,_ in enumerate(atom_coord)
                                       for j,_ in enumerate(atom_coord)
                                       if self.different_mols(i,j) and i<j])
        logger.info("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)

        # Intitial induced dipoles
        smearing_coeff = self.systems[0].Config.getfloat(
                            "induction","smearing_coeff")

        for i,_ in enumerate(atom_ele):
            self.induced_dip[i] = sum([atom_alpha_iso[i] *
                        self.interaction_permanent_multipoles(atom_coord[i],
                            atom_coord[j], atom_alpha_iso[i], atom_alpha_iso[j],
                            smearing_coeff, self.mtps_cart[j])
                            for j,_ in enumerate(atom_ele)
                            if self.different_mols(i,j)])
        logger.info("Initial induced dipoles [debye]:")
        for i,_ in enumerate(atom_ele):
            logger.info("Atom %d: %7.4f %7.4f %7.4f" % (i,
                            self.induced_dip[i][0] * constants.au2debye,
                            self.induced_dip[i][1] * constants.au2debye,
                            self.induced_dip[i][2] * constants.au2debye))
        # Self-consistent polarization
        mu_next = deepcopy(self.induced_dip)
        mu_prev = np.ones((len(atom_ele),3))
        cvg_threshld = self.systems[0].Config.getfloat(
                        "induction","convergence_thrshld")
        diff_init = np.linalg.norm(mu_next-mu_prev)
        counter = 0
        while np.linalg.norm(mu_next-mu_prev) > cvg_threshld:
            mu_prev = deepcopy(mu_next)
            for i,_ in enumerate(atom_ele):
                mu_next[i] = (1-omega)*mu_prev[i] + omega * (self.induced_dip[i] + sum(
                                [atom_alpha_iso[i] *
                                    product_smeared_ind_dip(atom_coord[i],
                                        atom_coord[j], self.cell,
                                        atom_alpha_iso[i], atom_alpha_iso[j],
                                        smearing_coeff, mu_prev[j])
                                    for j,_ in enumerate(atom_ele) if i != j]))
            counter += 1
            if np.linalg.norm(mu_next-mu_prev) > diff_init*10 or counter > 2000:
                logger.error("Can't converge self-consistent equations. Exiting.")
                exit(1)
            if counter % 50 == 0 and omega > 0.2:
                omega *= 0.8
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
        return self.energy_polarization + self.energy_shortranged


    def polarization_energy(self, smearing_coeff=None, stone_convention=False):
        """Compute induction energy"""
        self.convert_mtps_to_cartesian(stone_convention)
        # Populate Hirshfeld ratios of combined systems
        # self.hirshfelds = ...
        omega = self.systems[0].Config.getfloat("induction","omega")
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
                                Polarizability(self.sys_comb).get_pol_scaled())]

        # Short-range correction 
        self.energy_shortranged = 0.0 #-0.0480131753842 #comes from linear coefficient

        for i, c_i in enumerate(atom_coord):
            for j, c_j in enumerate(atom_coord):

                if (atom_typ[i] not in self.ele_ad):
                    print("Atom %s not parameterized!" % atom_typ[i]) 
                if (atom_typ[j] not in self.ele_ad):
                    print("Atom %s not parameterized!" % atom_typ[j]) 

                if (self.different_mols(i,j) and i < j):
                    #print "HERE", atom_typ[i], atom_typ[j]
                    fmbis = self.slater_mbis(
                    atom_coord[i], populations[i], valwidths[i], atom_typ[i],
                    atom_coord[j], populations[j], valwidths[j], atom_typ[j])
 
                    self.energy_shortranged +=  self.ind_sr[atom_typ[i]] * self.ind_sr[atom_typ[j]] * fmbis #/ pair_count[pairs_key.index(pair)]
        logger.info("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)
        # Intitial induced dipoles
        if smearing_coeff == None:
            smearing_coeff = self.systems[0].Config.getfloat(
                                "induction","smearing_coeff")

        for i,_ in enumerate(atom_ele):
            self.induced_dip[i] = sum([atom_alpha_iso[i] *
                        self.interaction_permanent_multipoles(atom_coord[i],
                            atom_coord[j], atom_alpha_iso[i], atom_alpha_iso[j],
                            smearing_coeff, self.mtps_cart[j])
                            for j,_ in enumerate(atom_ele)
                            if self.different_mols(i,j)])
        logger.info("Initial induced dipoles [debye]:")
        for i,_ in enumerate(atom_ele):
            logger.info("Atom %d: %7.4f %7.4f %7.4f" % (i,
                            self.induced_dip[i][0] * constants.au2debye,
                            self.induced_dip[i][1] * constants.au2debye,
                            self.induced_dip[i][2] * constants.au2debye))
        # Self-consistent polarization
        mu_next = deepcopy(self.induced_dip)
        mu_prev = np.ones((len(atom_ele),3))
        cvg_threshld = self.systems[0].Config.getfloat(
                        "induction","convergence_thrshld")
        diff_init = np.linalg.norm(mu_next-mu_prev)
        counter = 0
        while np.linalg.norm(mu_next-mu_prev) > cvg_threshld:
            mu_prev = deepcopy(mu_next)
            for i,_ in enumerate(atom_ele):
                mu_next[i] = (1-omega)*mu_prev[i] + omega * (self.induced_dip[i] + sum(
                                [atom_alpha_iso[i] *
                                    product_smeared_ind_dip(atom_coord[i],
                                        atom_coord[j], self.cell,
                                        atom_alpha_iso[i], atom_alpha_iso[j],
                                        smearing_coeff, mu_prev[j])
                                    for j,_ in enumerate(atom_ele) if i != j]))
            counter += 1
            if np.linalg.norm(mu_next-mu_prev) > diff_init*10 or counter > 2000:
                logger.error("Can't converge self-consistent equations. Exiting.")
                exit(1)
            if counter % 50 == 0 and omega > 0.2:
                omega *= 0.8
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

    def slater_mbis_v1(self, coord_i, N_i, v_i, typ_i, coord_j, N_j, v_j, typ_j):
        "Short-ranged induction model as described in Vandenbrande et al., JCTC, 13 (2017)"
        vec = self.cell.pbc_distance(coord_i, coord_j)
        rij = np.linalg.norm(vec)
        prefactor = 1.
        for typ in [typ_i, typ_j]:
            prefactor *= self.ind_sr[typ] if typ in self.ind_sr.keys() \
                                        else self.ind_sr[typ[0]]
      
        #print(typ_i, typ_j, rij, slater_mbis_funcform(rij, N_i, v_i, N_j, v_j))
        return prefactor * slater_mbis_funcform(rij, N_i, v_i, N_j, v_j)

    def slater_mbis(self, coord_i, N_i, v_i, typ_i, coord_j, N_j, v_j, typ_j):
        "Short-ranged induction model as described in Vandenbrande et al., JCTC, 13 (2017)"
        vec = self.cell.pbc_distance(coord_i, coord_j)
        rij = np.linalg.norm(vec)
        prefactor = 1.
        for typ in [typ_i, typ_j]:
            prefactor *= self.ind_sr[typ] if typ in self.ind_sr.keys() \
                                        else self.ind_sr[typ[0]]

        #print( typ_i, typ_j, rij, slater_mbis_funcform(rij, N_i, v_i, N_j, v_j))
        return prefactor * slater_mbis_funcform(rij, N_i, v_i, N_j, v_j)

@jit
def slater_mbis_funcform(rij, N_i, v_i, N_j, v_j):
    v_i2, v_j2 = v_i**2, v_j**2
    if abs(v_i-v_j) > 1e-3:
        # regular formula
        g0ab = -4*v_i2*v_j2/(v_i2-v_j2)**3
        g1ab = v_i/(v_i2-v_j2)**2
        g0ba = -4*v_j2*v_i2/(v_j2-v_i2)**3
        g1ba = v_j/(v_j2-v_i2)**2
        return N_i*N_j/(8*np.pi*rij) * \
                            ((g0ab+g1ab*rij)*np.exp(-rij/v_i) \
                            + (g0ba+g1ba*rij)*np.exp(-rij/v_j))
    else:
        rv = rij/v_i
        rv2, rv3, rv4 = rv**2, rv**3, rv**4
        exprv = np.exp(-rv)
        return N_i*N_j * \
            (1./(192*np.pi*v_i**3) * (3+3*rv+rv2) * exprv \
            + (v_j-v_i)/(384*v_i**4) * (-9-9*rv-2*rv2+rv3) * exprv \
            + (v_j-v_i)**2/(3840*v_i**5) * (90+90*rv+5*rv2-25*rv3+3*rv4) * exprv)

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

@jit
def convert_mtps_to_cartesian(mtp_sph, stone_convention):
    'Convert spherical MTPs to cartesian'
    # q={0,1,2} => 1+3+9 = 13 parameters
    mtp_cart = np.zeros((len(mtp_sph),13))
    if len(mtp_sph) == 0:
        logger.error("Multipoles not initialized!")
        print("Multipoles not initialized!")
        exit(1)
    for i in range(len(mtp_sph)):
        mtp_cart[i][0] = mtp_sph[i][0]
        mtp_cart[i][1] = mtp_sph[i][1]
        mtp_cart[i][2] = mtp_sph[i][2]
        mtp_cart[i][3] = mtp_sph[i][3]
        # Convert spherical quadrupole
        cart_quad = utils.spher_to_cart(
                        mtp_sph[i][4:9], stone_convention=stone_convention)
        # xx, xy, xz, yx, yy, yz, zx, zy, zz
        mtp_cart[i][4:13] = cart_quad.reshape((1,9))
    return mtp_cart
