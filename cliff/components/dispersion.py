#!/usr/bin/env python
#
# Dispersion class. Compute many-body dispersion.
#

from cliff.atomic_properties.polarizability import Polarizability, cutoff
from functools import reduce
import operator
import numpy as np
import cliff.helpers.constants as constants
import math
import logging

# Set logger
logger = logging.getLogger(__name__)

class Dispersion():
    'Dispersion class. Computes many-body dispersion'

    def __init__(self, options, _system, cell):
        logger.setLevel(options.logger_level)

        self.systems = [_system]
        self.energy = 0.0
        self.cell = cell
        self.radius = options.disp_radius
        self.beta = options.disp_beta
        self.scs_cutoff = options.pol_scs_cutoff
        self.pol_exponent = options.pol_exponent


    def add_system(self, sys):
        self.systems.append(sys)
        

    def compute_dispersion(self, method='MBD', hirsh=None):

        # Driver for dispersion computations
        if method == 'MBD':
            if hirsh == None:
                raise Exception("Must pass Hirshfeld model for MBD method")
            dimer = reduce(operator.add, self.systems)
            hirsh.predict_mol(dimer)

            disp = 0.0
            for i, mol in enumerate([dimer] + self.systems):
                fac = 1.0 if i == 0 else -1.0
                pol = Polarizability(self.scs_cutoff,self.pol_exponent,mol)
                pol.compute_freq_pol()
                #compute anisotropic characteristic frequencies
                pol.compute_freq_scaled_anisotropic()
                #execute MBD protocol
                disp += fac * self.mbd_protocol(pol,None,None,None)
            return disp
            
            

    def mbd_protocol(self, pol, radius=None, beta=None, scs_cutoff=None):
        'Compute many-body dispersion and molecular polarizability'
        size = (3*pol.num_atoms,3*pol.num_atoms)
        int_mat = np.zeros(size)
        c_mat = np.zeros(size)

        ele_list = ['H', 'C', 'O', 'N', 'S', 'Cl', 'F']
    
        if radius != None:
            self.radius = radius
        if beta != None:
            self.beta = beta
        if scs_cutoff != None:
            self.scs_cutoff = scs_cutoff

        #print("Using parameters: %f, %f, and %f" %(radius,beta,scs_cutoff))
        for ati in range(pol.num_atoms):
            for atj in range(pol.num_atoms):
                if ati == atj:
                    for ri in range(3):
                        int_mat[3*ati+ri,3*ati+ri] = pol.freq_scaled[ati]**2
                        c_mat[3*ati+ri,3*ati+ri]   = 1./(
                            pol.pol_scaled[ati] * \
                            pol.freq_scaled_vec[ati][ri]**2)
                else:
                    for ri in range(3):
                        for rj in range(3):
                            rij = self.cell.pbc_distance(pol.system.coords[ati], \
                                pol.system.coords[atj])*constants.a2b
                            rijn = np.linalg.norm(rij)
                            # Kronecker delta between two coordinates
                            delta_ab = 1.0 if ri == rj else 0.0
                            # Compute effective width sigma
                            sigma = self.radius * (pol.radius_vdw(ati) + \
                                pol.radius_vdw(atj))
                            frac = (rijn/sigma)**self.beta
                            expf = math.exp(-frac)
                            int_mat[3*ati+ri,3*atj+rj] = \
                                pol.freq_scaled[ati] * \
                                pol.freq_scaled[atj] * \
                                math.sqrt(pol.pol_scaled[ati] * \
                                    pol.pol_scaled[atj]) * (
                                    (-3.*rij[ri]*rij[rj] +rijn**2*delta_ab) \
                                    /rijn**5 * cutoff(rijn, sigma, pol.scs_cutoff) \
                                    * (1 - expf - self.beta*frac*expf) + \
                                    (self.beta*frac+1-self.beta)*self.beta*frac* \
                                    rij[ri]*rij[rj]/rijn**5*expf )
        # Compute eigenvalues
        eigvals,eigvecs = np.linalg.eigh(int_mat)
        for i in range(3*pol.num_atoms):
            eigvecs[:,i] /= math.sqrt( np.dot( np.dot(
                eigvecs.transpose()[i],c_mat),
                eigvecs.transpose()[i]))
        # Group eigenvectors into components
        aggr = sum(eigvecs[:,i]*eigvecs[:,i]/eigvals[i] for i in \
            range(len(eigvecs)))
        amol = np.zeros(3)
        for i in range(pol.num_atoms):
            for j in range(3):
                amol[j] += aggr[3*i+j]
        # print aggr.reshape((self.num_atoms,3))
        # print np.array([sum(aggr.reshape((self.num_atoms,3))[i][j] for j in range(3))/3. for i in range(self.num_atoms)])
        # Molecular polarizability
        pol.pol_mol_iso = sum(amol)/3.
        logger.info("isotropic molecular polarizability: %7.4f" % \
            pol.pol_mol_iso)
        pol.pol_mol_vec = np.array([amol[0],amol[1],amol[2]])
        logger.info("molecular polarizability tensor: %7.4f %7.4f %7.4f" % \
            (pol.pol_mol_vec[0],pol.pol_mol_vec[1],pol.pol_mol_vec[2]))
        # Fractional anisotropy
        pol.pol_mol_fracaniso = math.sqrt(0.5 * ((amol[0]-amol[1])**2 + \
            (amol[0]-amol[2])**2 + (amol[1]-amol[2])**2) \
            / (amol[0]**2 + amol[1]**2 + amol[2]**2))
        logger.info("Fractional anisotropy: %7.4f" % pol.pol_mol_fracaniso)
        # print eigvals
        # print self.freq_scaled, 3.*sum(self.freq_scaled)
        # print sum([math.sqrt(eigvals[i]) for i in range(len(eigvals))]), \
        #     3.*sum(self.freq_scaled), sum([math.sqrt(eigvals[i]) for i in range(len(eigvals))]) \
        #     - 3.*sum(self.freq_scaled)
        energy = .5*(sum([math.sqrt(eigvals[i]) for i in \
            range(len(eigvals))]) - \
            3*sum(pol.freq_scaled)) * constants.au2kcalmol
        logger.info("energy: %7.4f kcal/mol" % self.energy)
        return energy
