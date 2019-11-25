#!/usr/bin/env python
#
# Dispersion class. Compute many-body dispersion.
#

from cliff.atomic_properties.polarizability import Polarizability, cutoff
import numpy as np
import cliff.helpers.constants as constants
import math
import logging

# Set logger
logger = logging.getLogger(__name__)

class Dispersion(Polarizability):
    'Dispersion class. Computes many-body dispersion'

    def __init__(self, options, _system, cell):
        Polarizability.__init__(self,options, _system)
        logger.setLevel(options.logger_level)
        self.energy = 0.0
        self.cell = cell
        self.radius = options.disp_radius
        self.beta = options.disp_beta
        self.scs_cutoff = options.pol_scs_cutoff

    def mbd_protocol(self, radius=None, beta=None, scs_cutoff=None):
        'Compute many-body dispersion and molecular polarizability'
        size = (3*self.num_atoms,3*self.num_atoms)
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
        for ati in range(self.num_atoms):
            for atj in range(self.num_atoms):
                if ati == atj:
                    for ri in range(3):
                        int_mat[3*ati+ri,3*ati+ri] = self.freq_scaled[ati]**2
                        c_mat[3*ati+ri,3*ati+ri]   = 1./(
                            self.pol_scaled[ati] * \
                            self.freq_scaled_vec[ati][ri]**2)
                else:
                    for ri in range(3):
                        for rj in range(3):
                            rij = self.cell.pbc_distance(self.system.coords[ati], \
                                self.system.coords[atj])*constants.a2b
                            rijn = np.linalg.norm(rij)
                            # Kronecker delta between two coordinates
                            delta_ab = 1.0 if ri == rj else 0.0
                            # Compute effective width sigma
                            sigma = self.radius * (self.radius_vdw(ati) + \
                                self.radius_vdw(atj))
                            frac = (rijn/sigma)**self.beta
                            expf = math.exp(-frac)
                            int_mat[3*ati+ri,3*atj+rj] = \
                                self.freq_scaled[ati] * \
                                self.freq_scaled[atj] * \
                                math.sqrt(self.pol_scaled[ati] * \
                                    self.pol_scaled[atj]) * (
                                    (-3.*rij[ri]*rij[rj] +rijn**2*delta_ab) \
                                    /rijn**5 * cutoff(rijn, sigma, self.scs_cutoff) \
                                    * (1 - expf - self.beta*frac*expf) + \
                                    (self.beta*frac+1-self.beta)*self.beta*frac* \
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
        for i in range(self.num_atoms):
            for j in range(3):
                amol[j] += aggr[3*i+j]
        # print aggr.reshape((self.num_atoms,3))
        # print np.array([sum(aggr.reshape((self.num_atoms,3))[i][j] for j in range(3))/3. for i in range(self.num_atoms)])
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
        # print eigvals
        # print self.freq_scaled, 3.*sum(self.freq_scaled)
        # print sum([math.sqrt(eigvals[i]) for i in range(len(eigvals))]), \
        #     3.*sum(self.freq_scaled), sum([math.sqrt(eigvals[i]) for i in range(len(eigvals))]) \
        #     - 3.*sum(self.freq_scaled)
        self.energy = .5*(sum([math.sqrt(eigvals[i]) for i in \
            range(len(eigvals))]) - \
            3*sum(self.freq_scaled)) * constants.au2kcalmol
        logger.info("energy: %7.4f kcal/mol" % self.energy)
        return None
