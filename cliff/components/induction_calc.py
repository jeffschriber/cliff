#!/usr/bin/env python

import numpy as np
from cliff.helpers.system import System
from cliff.components.electrostatics import Electrostatics, interaction_tensor
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.polarizability import Polarizability
#from cliff.helpers.cell import Cell
from numpy import exp
import cliff.helpers.cell as Cell
from copy import deepcopy
import logging
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
import time


class InductionCalc(Electrostatics):

    def __init__(self, options, sys, cell, ind_sr=None, hirshfeld_pred="krr", v1=False):
        # Set logger
        name = options.name
        self.logger = options.logger

        Electrostatics.__init__(self,options,sys,cell)
        self.logger.setLevel(options.logger_level)
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
        v_widths = []
        atom_alpha_iso = []
        ind_params = []
        induced_dip = []
        
        for sys in self.systems:
            atom_coord.append([crd*constants.a2b for crd in sys.coords])
            atom_ele.append([ele for ele in sys.elements])
            atom_typ.append([typ for typ in sys.atom_types])
            v_widths.append([v for v in sys.valence_widths])
            induced_dip.append(np.zeros((len(sys.elements),3)))

            # Atomic polarizabilities
            atom_alpha_iso.append([alpha for alpha in Polarizability(options.name, self.logger,self.scs_cutoff,self.pol_exponent,sys).get_pol_scaled()])
            ind_params.append([self.ind_sr[i] for i in sys.atom_types]) 

        # Compute the short-range correction
        # This is done with U_aU_b*S(a,b,r),
        # which is the same formalism as exchange
        self.energy_shortranged = 0.0 
        start_sr = time.time()
        for s1 in range(nsys):
            for s2 in range(s1+1, nsys):
                r = utils.build_r(atom_coord[s1], atom_coord[s2], self.cell)
                r1 = utils.build_r(atom_coord[s1], atom_coord[s1], self.cell)
                r2 = utils.build_r(atom_coord[s2], atom_coord[s2], self.cell)
                ovp = utils.slater_ovp_mat(r,v_widths[s1],v_widths[s2])
                self.energy_shortranged = np.dot(ind_params[s1], np.matmul(ovp,ind_params[s2]))

                u = self.build_u(r, atom_alpha_iso[s1], atom_alpha_iso[s2])
                u_1 = self.build_u(r1, atom_alpha_iso[s1], atom_alpha_iso[s1])
                u_2 = self.build_u(r2, atom_alpha_iso[s2], atom_alpha_iso[s2])

        self.energy_shortranged *= constants.au2kcalmol
        end_sr = time.time()
        #logger.info("short-range: %6.3f" % (end_sr - start_sr))
        #print("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)
        #logger.info("Induction energy: %7.4f kcal/mol" % self.energy_shortranged)

        ###  Compute the induction term using Thole's formalism

        start_pol = time.time()
        if smearing_coeff != None:
            self.smearing_coeff = smearing_coeff 

        # Compute the initial induced dipoles for each atom
        # These correspond to: mu_i = alpha_i * sum_j(T^ij_a M_j) 
        # where we damp T using Thole's formalism

        # fix the charges
        for s,sys in enumerate(self.systems):
            for i in range(sys.num_atoms):
                self.mtps_cart[s][i][0] += constants.atomic_number[atom_ele[s][i]]

        # Gather intramonomer dipole interaction tensors  for later too
        T_dd_1_self = [] 
        T_dd_2_self = [] 

        # grab the dipole-dipole part for later
        T_dd_1 = []
        T_dd_2 = []

        for s1 in range(nsys):
            mtp1 = self.mtps_cart[s1]
            for s2 in range(s1+1, nsys):
                mtp2 = self.mtps_cart[s2]
            
                for i,atom in enumerate(atom_ele[s1]):
                    T_1 = self.build_int_tensor(atom_coord[s1][i], atom_coord[s2], u[i,:], self.smearing_coeff)
                    induced_dip[s1][i] = np.einsum("ijk,ki->j",T_1,mtp2.T) * atom_alpha_iso[s1][i]
                    T_dd_1.append(T_1[:,:,1:4])

                    T_dd_1_self.append(self.build_self_dip_int_tensor(atom_coord[s1][i], atom_coord[s1], u_1[i,:], self.smearing_coeff))


                for i,atom in enumerate(atom_ele[s2]):
                    T_2 = self.build_int_tensor(atom_coord[s2][i], atom_coord[s1], u[:,i], self.smearing_coeff)
                    induced_dip[s2][i] = np.einsum("ijk,ki->j",T_2,mtp1.T) * atom_alpha_iso[s2][i]
                    T_dd_2.append(T_2[:,:,1:4])

                    T_dd_2_self.append(self.build_self_dip_int_tensor(atom_coord[s2][i], atom_coord[s2], u_2[i,:], self.smearing_coeff))

        end_init = time.time()

#        logger.info("Initial induced dipole:")
#        logger.info("Mol 1")
#        for vec in induced_dip[0]:
#            tmp_vec = vec * constants.au2debye
#            logger.info("%9.6f  %9.6f  %9.6f" %(tmp_vec[0],tmp_vec[1],tmp_vec[2]))
#        logger.info("Mol 2")
#        for vec in induced_dip[1]:
#            tmp_vec = vec * constants.au2debye
#            logger.info("%9.6f  %9.6f  %9.6f" %(tmp_vec[0],tmp_vec[1],tmp_vec[2]))

        #logger.info("init: %6.3f" % (end_init - start_pol))
        # Self-consistent polarization
        mu_next = np.copy(induced_dip)
        #mu_prev = np.zeros(np.shape(mu_next))
        #mu_prev = np.zeros(np.shape(induced_dip))
        mu_prev = []
        for i in induced_dip:
            mu_prev.append(np.zeros(np.shape(i)))

       # diff_init = np.linalg.norm(mu_next-mu_prev)
        diff_init = 0.0
        for n in range(nsys):
            diff_init += np.linalg.norm(mu_next[n]-mu_prev[n])
    
        counter = 0

        # Compute the induced dipoles
        # We already have the interaction tensors
        diff = diff_init
        while diff > self.conv:
            mu_prev = np.copy(mu_next)
            mu_next = (1.0 - self.omega) * mu_prev
            for s1 in range(nsys):
                mp_1 = mu_prev[s1]
                for s2 in range(s1+1,nsys):
                    mp_2 = mu_prev[s2]

                    tmp = np.einsum("nijk,ki,n->nj", T_dd_1, mp_2.T, atom_alpha_iso[s1]) + \
                          np.einsum("nijk,ki,n->nj", T_dd_1_self, mp_1.T, atom_alpha_iso[s1])
                    mu_next[s1] += (tmp + induced_dip[s1])*self.omega

                    tmp = np.einsum("nijk,ki,n->nj", T_dd_2, mp_1.T, atom_alpha_iso[s2]) + \
                          np.einsum("nijk,ki,n->nj", T_dd_2_self, mp_2.T, atom_alpha_iso[s2])
                    mu_next[s2] += (tmp + induced_dip[s2])*self.omega

            counter += 1
            diff = 0.0
            for n in range(nsys):
                diff += np.linalg.norm(mu_next[n]-mu_prev[n])
            if diff > diff_init*10 or counter > 2000:
                logger.error("Can't converge self-consistent equations. Exiting.")
                exit(1)
            if counter % 50 == 0 and self.omega > 0.2:
                self.omega *= 0.8

        #self.induced_dip = np.zeros(np.shape(self.mtps_cart))
        self.induced_dip = []
        for i in self.mtps_cart:
            self.induced_dip.append(np.zeros(np.shape(i)))


#        logger.debug("Converged induced dipoles [debye]:")
        for s in range(nsys):
 #           logger.info("Mol %d" % s)
            for n,vec in enumerate(mu_next[s]):
                self.induced_dip[s][n][1:4] = vec            
#                tmp_vec = vec * constants.au2debye
#                logger.info("%9.6f  %9.6f  %9.6f" %(tmp_vec[0],tmp_vec[1],tmp_vec[2]))

        self.energy_polarization = 0.0
        for s1 in range(nsys):
            for s2 in range(s1+1,nsys):
                for atom1 in range(len(atom_ele[s1])):
                    crdi = atom_coord[s1][atom1]        
                    mi1 = self.mtps_cart[s1][atom1,:] 
                    mind_1 = self.induced_dip[s1][atom1,:]
                    for atom2 in range(len(atom_ele[s2])):
                        crdj = atom_coord[s2][atom2]        
                        mj2 = self.mtps_cart[s2][atom2,:] 
                        mind_2 = self.induced_dip[s2][atom2,:]
                        T = interaction_tensor(crdi, crdj,self.cell)

                        self.energy_polarization += np.dot(mind_1.T,np.dot(T, mj2)) + np.dot(mi1.T,np.dot(T, mind_2))
                        

        self.energy_polarization *= 0.5 * constants.au2kcalmol 


        end_pol = time.time()
        #logger.debug("Polarization energy: %7.4f kcal/mol" % self.energy_polarization)
        #print("Polarization energy: %7.4f kcal/mol" % self.energy_polarization)
        #print "Short range", self.energy_shortranged
       # print(str(self.sys_comb)[:-1],self.energy_polarization , self.energy_shortranged)
        return self.energy_polarization - self.energy_shortranged

    def build_u(self,r, a1, a2): 

        u = np.copy(r) 
        a = np.outer(a1,a2)
        a = np.power(a, (1/6))
        u = np.divide(u,a)
        return u

    def build_self_dip_int_tensor(self, coord1, coord2, u_i, smear):

        r = np.zeros((len(coord2),3))
        xyz = np.zeros((len(coord2),3))
        exp = np.zeros((len(coord2),3))
        l3 = np.zeros((len(coord2),3))
        l5 = np.zeros((len(coord2),3))

        T = np.zeros([len(coord2),3,3])

        ref = 0

        for n, coord in enumerate(coord2):
            vec = self.cell.pbc_distance(coord1, coord)
            xyz[n] = np.asarray(vec)
            r[n] = np.asarray(np.linalg.norm(vec))

            if r[n][0] < 1e-8:
                ref = n 
                r[n] = np.asarray(1.0)

            exp = np.multiply(np.power(u_i[n],3.0),-smear)
            l3[n] = np.asarray(1.0 - np.exp(exp)) 
            l5[n] = np.asarray(1.0 - (1 - exp)*np.exp(exp) )

        # dipole-dipole
        r5 = np.power(r,-5.0)
        for n in range(3):
            dd = np.copy(xyz)
            for m in range(3):
                dd[:,m] = np.multiply(dd[:,m], 3*xyz[:,n])

            dd = np.multiply(dd, r5)
            T[:,:,n] = np.multiply(dd,l5)
                
            # diagonal term
            T[:,n,n] -= np.multiply(l3[:,0], np.power(r[:,0],-3.0))
        
        # set ref to zero
        T[ref,:,:] = np.asarray(0.0)

        return T

    def build_int_tensor(self, coord1, coord2, u_i, smear):
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
            vec = self.cell.pbc_distance(coord1, coord)
            xyz[n] = np.asarray(vec)
            r[n] = np.asarray(np.linalg.norm(vec))

            exp = np.multiply(np.power(u_i[n],3.0),-smear)
            l3[n] = np.asarray(1.0 - np.exp(exp)) 
            l5[n] = np.asarray(1.0 - (1 - exp)*np.exp(exp) )
            l7[n] = np.asarray(1.0 - (1.0 - exp + 0.6*exp*exp)*np.exp(exp))
        #T = np.zeros([3,13,r.shape[0]])
        T = np.zeros([r.shape[0],3,13])

        # dipole-charge
        dc = np.copy(xyz)
        dc = np.multiply(dc, np.power(r,-3))    
        dc = np.multiply(dc,-l3)
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
                    dq[:,p] += np.multiply(np.multiply(num,l5[:,0]), r5[:,0])

                T[:,:,4 + m*3 + n] = dq 
                
     #   print(T)    
        return T

