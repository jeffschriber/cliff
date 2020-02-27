#!/usr/bin/env python

import numpy as np
from cliff.helpers.system import System
from cliff.components.electrostatics import Electrostatics, interaction_tensor
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.polarizability import Polarizability
from cliff.helpers.cell import Cell
from numpy import exp
from copy import deepcopy
import logging
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
from numba import jit
import time

# Set logger
logger = logging.getLogger(__name__)
fh = logging.FileHandler('output.log')
logger.addHandler(fh)

class InductionCalc(Electrostatics):

    def __init__(self, options, sys, cell, ind_sr=None, hirshfeld_pred="krr", v1=False):
        Electrostatics.__init__(self,options,sys,cell)
        logger.setLevel(options.logger_level)
        self.cell = cell
        self.induced_dip = []
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
        Electrostatics.add_system(self, sys)
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
        nsys = len(self.systems)
        
        atom_coord = []    
        atom_ele = []    
        atom_typ = []    
        pops = []
        v_widths = []
        atom_alpha_iso = []
        ind_params = []
        induced_dip = []
        
        for sys in self.systems:
            atom_coord.append([crd for crd in sys.coords])
            atom_ele.append([ele for ele in sys.elements])
            atom_typ.append([typ for typ in sys.atom_types])

            pops.append([p for p in sys.populations])
            v_widths.append([v/constants.a2b for v in sys.valence_widths])

            induced_dip.append(np.zeros((len(sys.elements),3)))

            # Atomic polarizabilities
            atom_alpha_iso.append([alpha for alpha in Polarizability(self.scs_cutoff,self.pol_exponent,sys).get_pol_scaled()])
    
            ind_params.append([self.ind_sr[i] for i in sys.atom_types]) 

        # Compute the short-range correction
        # This is done with U_aU_b*S(a,b,r),
        # which is the same formalism as exchange
        self.energy_shortranged = 0.0 

        r_list = []
        u_list = []

        start_sr = time.time()
        for s1 in range(nsys):
            for s2 in range(s1+1, nsys):
                r = utils.build_r(atom_coord[s1], atom_coord[s2], self.cell)
                ovp = utils.slater_ovp_mat(r,v_widths[s1],pops[s1],v_widths[s2], pops[s2])

                for ov in ovp.flatten():
                    logger.info("%13.12f"%ov)

                self.energy_shortranged += np.dot(ind_params[s1], np.matmul(ovp,ind_params[s2]))
                #r_list.append(r)

                u = self.build_u(r, atom_alpha_iso[s1], atom_alpha_iso[s2])

        end_sr = time.time()
        logger.info("short-range: %6.3f" % (end_sr - start_sr))
        print("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)
        logger.info("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)

        ###  Compute the induction term using Thole's formalism

        start_pol = time.time()
        if smearing_coeff != None:
            self.smearing_coeff = smearing_coeff 

        # Compute the initial induced dipoles for each atom
        # These correspond to: mu_i = alpha_i * sum_j(T^ij_a M_j) 
        # where we damp T using Thole's formalism
        for s1 in range(nsys):
            mtp1 = self.mtps_cart[s1]
            for s2 in range(s1+1, nsys):
                mtp2 = self.mtps_cart[s2]
                #r = r_list[s1*nsys + s2 -1] * constants.a2b  
            
                for i,atom in enumerate(atom_ele[s1]):
                    T_1 = self.build_int_tensor(atom_coord[s1][i], atom_coord[s2], u[i,:], self.smearing_coeff)

                    induced_dip[s1][i] = np.einsum("ijk,ki->j",T_1,mtp2.T) * atom_alpha_iso[s1][i]

                for i,atom in enumerate(atom_ele[s2]):
                    T_2 = self.build_int_tensor(atom_coord[s2][i], atom_coord[s1], u[i,:], self.smearing_coeff)
                    induced_dip[s2][i] = np.einsum("ijk,ki->j",T_2,mtp1.T) * atom_alpha_iso[s2][i]

                #int_perm_mult = self.interaction_permanent_multipoles(r, 
                #                    atom_alpha_iso[s1],atom_alpha_iso[s2],
                #                    self.smearing_coeff, self.mtps_cart[s2])
        logger.info("Initial induced dipole:")
        logger.info("Mol 1")
        for vec in induced_dip[0]:
            logger.info("%9.6f  %9.6f  %9.6f" %(vec[0],vec[1],vec[2]))
        logger.info("Mol 2")
        for vec in induced_dip[0]:
            logger.info("%9.6f  %9.6f  %9.6f" %(vec[0],vec[1],vec[2]))

        end_init = time.time()
        logger.info("init: %6.3f" % (end_init - start_pol))
        exit()
        # Self-consistent polarization
        mu_next = deepcopy(self.induced_dip)
        mu_prev = np.ones((len(atom_ele),3))
        diff_init = np.linalg.norm(mu_next-mu_prev)
        counter = 0

        # build interaction tensor out of loop


        # compute the induced dipoles
        while np.linalg.norm(mu_next-mu_prev) > self.conv:
            mu_prev = deepcopy(mu_next)
            for i,_ in enumerate(atom_ele):
                mu_next[i] = (1-self.omega)*mu_prev[i] + self.omega * (int_perm_mult + sum(
                                [atom_alpha_iso[i] *
                                    product_smeared_ind_dip(atom_coord[i],
                                        atom_coord[j], self.cell,
                                        atom_alpha_iso[i], atom_alpha_iso[j],
                                        self.smearing_coeff, mu_prev[j])
                                    for j,_ in enumerate(atom_ele) if i != j]))
            counter += 1
            if np.linalg.norm(mu_next-mu_prev) > diff_init*10 or counter > 2000:
                logger.error("Can't converge self-consistent equations. Exiting.")
                exit(1)
            if counter % 50 == 0 and self.omega > 0.2:
                self.omega *= 0.8
        self.induced_dip = np.zeros((len(atom_ele),13))
        logger.debug("Converged induced dipoles [debye]:")


        for i,_ in enumerate(atom_ele):
            logger.debug("Atom %d: %7.4f %7.4f %7.4f" % (i,
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

        end_pol = time.time()
        print("Pol: %6.3f" % (end_pol - start_pol))
        logger.debug("Polarization energy: %7.4f kcal/mol" % self.energy_polarization)
        #print("Polarization energy", self.energy_polarization)
        #print "Short range", self.energy_shortranged
        #print self.energy_polarization , self.energy_shortranged
        return self.energy_polarization - self.energy_shortranged

    def interaction_permanent_multipoles(self, r, at_pol1, at_pol2,
        smearing, mtp_perm):
        """
        Returns product of smeared interaction tensor with permanent multipoles.
        Corresponds to the external field.
    
        Computes for all 
        """

        interac = np.zeros(3)
        # Permanent charge contribution + monopole charge penetration
        charge = mtp_perm[0]


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


    def build_u(self,r, a1, a2): 

        u = np.copy(r) 
        a = np.outer(a1,a2)
        a = np.power(a, (1/6))
        u = np.divide(u,a)
        return u

    def build_int_tensor(self, coord1, coord2, u, smear):
        """
        Returns natom x 3 x 13 interaction tensor
        Interaction of atom i with all atoms in other system
        
        @params
    
        coords, array
            XYZ coords of ref element in reference system
        
        r, array
            Distance vector of element i in sys1 and all in 
    
        u, array
            Effective distance matrix weighted by element polarizabilities,
            same dim as r
    
        smear, float
            Smearing coefficient in Thole model 
        """
        r = np.zeros((len(coord2),3))
        xyz = np.zeros((len(coord2),3))
        exp = np.zeros((len(coord2),3))
        l3 = np.zeros((len(coord2),3))
        l5 = np.zeros((len(coord2),3))
        l7 = np.zeros((len(coord2),3))

        for n, coord in enumerate(coord2):
            vec = constants.a2b*(self.cell.pbc_distance(coord1, coord))
            xyz[n] = np.asarray(vec)
            r[n] = np.asarray(np.linalg.norm(vec))

            exp = np.multiply(np.power(u[n],3.0),-smear)
            l3[n] = np.asarray(1.0 - np.exp(exp)) 
            l5[n] = np.asarray(1.0 - (1 - exp)*np.exp(exp) )
            l7[n] = np.asarray(1.0 - (1.0 - exp - 0.6*exp**2)*np.exp(exp))

        #T = np.zeros([3,13,r.shape[0]])
        T = np.zeros([r.shape[0],3,13])

        # dipole-charge
        dc = np.copy(xyz)
        dc = np.multiply(dc, np.power(r,-3))    
        dc = np.multiply(dc,l3)
        T[:,:,0] = dc 

        # dipole-dipole
        r5 = np.power(r,-5.0)
        for n in range(3):
            dd = np.copy(xyz)
            for m in range(3):
                dd[:,m] = np.multiply(dd[:,m], 3*xyz[:,n])

            dd = np.multiply(dd, r5)
            T[:,:,n+1] = np.multiply(dd,l5)
                
            # diagonal term
            T[:,n,n+1] -= np.multiply(l3[:,0], np.power(r[:,0],-3.0))

        # dipole-quadripole
        r7 = np.power(r,-7.0)
        for m in range(3):
            for n in range(3):
                dq = np.multiply(xyz,r7) * -15.0
                dq = np.multiply(dq,l7)
                for p in range(3):
                    dq[:,p] = np.asarray(np.multiply(dq[:,p], np.multiply(xyz[:,m],xyz[:,n])))
                    
                    num = np.zeros(len(coord2))
                    if m == n:
                        num += xyz[:,p]
                    if m == p:
                        num += xyz[:,n]
                    if n == p:
                        num += xyz[:,m]
                    num *= 3
                    dq[:,p] += np.multiply(np.multiply(num,l7[:,0]), r7[:,0])

                T[:,:,4 + m*3 + n] = dq 
                
     #   print(T)    
        return T


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

#@jit
def interaction_tensor_first(vec, at_pol1, at_pol2, smearing, dir1):
    """
    Returns smeared dipole-charge interaction tensor using Thole formalism.
    """
    r = np.linalg.norm(vec)
    u = r/(at_pol1*at_pol2)**(1/6.)
    ri3 = 1./r**3
    ret =  -(1.-np.exp(-smearing*u**3))*vec[dir1]*ri3
    return ret

#@jit
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

#@jit
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
    ret =  - lambda7*15*vec[dir1]*vec[dir2]*vec[dir3]*ri7 \
            + lambda5*3*(coeff1+coeff2+coeff3)*ri5
    return ret


