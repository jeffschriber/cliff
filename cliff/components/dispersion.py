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

        self.method = options.disp_method
        self.systems = [_system]
        self.energy = 0.0
        self.cell = cell
        self.radius = options.disp_radius
        self.beta = options.disp_beta
        self.scs_cutoff = options.pol_scs_cutoff
        self.pol_exponent = options.pol_exponent

        self.disp_coeffs = options.disp_coeffs

    def add_system(self, sys):
        self.systems.append(sys)
        
    def compute_dispersion(self, hirsh=None):

        # Driver for dispersion computations
        if self.method == "TT":
            
            disp = self.compute_tang_toennies()

            return disp * constants.au2kcalmol

        if self.method == 'MBD':
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
            
    def compute_tang_toennies(self):
        '''
        Computes total dispersion energy using Tang-Toennies damping.
        Uses prefactor from Van Vleet 2018
        '''        
        # assume two systems
        sys_i = self.systems[0]                    
        sys_j = self.systems[1]                    

        # compute c6 coefficients
        c6_ab = self.compute_c6_coeffs()
        c8_ab = self.compute_c8_coeffs(c6_ab)
        #cdict = {6:c6_ab,8:c8_ab}  

        #for A, ele_A in enumerate(sys_i.atom_types):
        #    for B, ele_B in enumerate(sys_j.atom_types):
        #        c8_ab[A][B] *= self.disp_coeffs[ele_A]*self.disp_coeffs[ele_B]
        c10_ab = self.compute_c10_coeffs(c6_ab, c8_ab)


        disp = 0.0
        
        for A, ele_A in enumerate(sys_i.atom_types):
            # valence decay rates
            b_A = 1.0 / (sys_i.valence_widths[A])
            for B, ele_B in enumerate(sys_j.atom_types):
                b_B = 1.0/(sys_j.valence_widths[B])

                # get interatomic distance
                coord_A = sys_i.coords[A]
                coord_B = sys_j.coords[B]
                vec = self.cell.pbc_distance(coord_A,coord_B) * constants.a2b
                rAB = np.linalg.norm(vec)

                # use combining rule
                b_AB = np.sqrt(b_A*b_B)
                
                #for n in [6,8]:
                #    fn = self.compute_tt_damping(n, rAB, b_AB)
                #    cn = cdict[n]
                #         
                #    dij += fn * cn[A][B] / (rAB**n)   
                
                f6 = self.compute_tt_damping(6, rAB, b_AB)
                f8 = self.compute_tt_damping(8, rAB, b_AB)
                f10 = self.compute_tt_damping(10, rAB, b_AB)

                #disp += self.scale6 * f6*c6_ab[A][B]/(rAB**6.0) 
                #disp += self.scale8 * f8*c8_ab[A][B]/(rAB**8.0)
                disp -= f6*c6_ab[A][B]/(rAB**6.0) 
                #disp -= self.disp_coeffs[ele_A]*self.disp_coeffs[ele_B] * f8*c8_ab[A][B]/(rAB**8.0)
                disp -= (f8*c8_ab[A][B]/(rAB**8.0) + f10*c10_ab[A][B]/(rAB**10.0)) * self.disp_coeffs[ele_A]*self.disp_coeffs[ele_B]

        return disp


    def compute_tt_damping(self, n, rAB, b_AB):
        '''
        Computes Tang--Toennies damping for dispersion, thanks to MVV
        
        @params:
        
        n: Order of damping funcion, usually 6 or 8
    
        rab: The interatomic distance in au

        b_AB: the Bab parameter, computed as square root of product of
              the inverse of the valence widths for atoms A and B

        '''
        # compute x
        b2 = b_AB*b_AB

        x = b_AB*rAB - ((2.0*b2*rAB + 3*b_AB)*rAB / (b2*rAB*rAB + 3.0*b_AB*rAB + 3.0))

        # Compute damping function
        x_sum = 1.0
        # so far we are hard-coding use of C6 and C8 only
        for k in range(1, n+1):
            x_sum += (x**k)/math.factorial(k)

        return 1.0 - np.exp(-x)*x_sum


    def compute_c6_coeffs(self):

        nsys = len(self.systems)

        if nsys <= 1:
            raise Exception("Need at least two monomers")
        
        # assume two systems
        sys_i = self.systems[0]                    
        sys_j = self.systems[1]                    

        C6_AB = np.zeros([len(sys_i.elements), len(sys_j.elements)])

        # hirshfeld ratios
        hi = sys_i.hirshfeld_ratios
        hj = sys_j.hirshfeld_ratios

        for A,ele_A in enumerate(sys_i.elements):
            # get effective C6s from free-atom C6s
            c6_AA = constants.csix_free[ele_A]*hi[A]*hi[A]

            # get effective atomic polarizabilities
            a_A = hi[A] * constants.pol_free[ele_A]

            for B,ele_B in enumerate(sys_j.elements):
                c6_BB = constants.csix_free[ele_B]*hj[B]*hj[B]
                a_B = hj[B] * constants.pol_free[ele_B]
                
                C6_AB[A][B] = (2.0 * c6_AA * c6_BB) / ((a_B/a_A)*c6_AA + (a_A/a_B)*c6_BB)

        return C6_AB

    def compute_c8_coeffs(self, C6_AB):
        '''
        Computes C8 coefficients using the Starkschall recursion relation
        '''

        # 1. First copy the C6s in the C8s
        C8_AB = np.copy(C6_AB)

        # 2. Grab systems
        sys_i = self.systems[0]                    
        sys_j = self.systems[1]                    
        hi = sys_i.hirshfeld_ratios
        hj = sys_j.hirshfeld_ratios
        
        for A, ele_A in enumerate(sys_i.elements):
            # 3. For each atom, get free-atom  <r2> and <r4>
            r2_A = constants.atomic_r2[ele_A]
            r4_A = constants.atomic_r4[ele_A]
            r42A = r4_A / r2_A
            for B, ele_B in enumerate(sys_j.elements):
                # 3. For each atom, get free-atom  <r2> and <r4>
                r2_B = constants.atomic_r2[ele_B] 
                r4_B = constants.atomic_r4[ele_B]
                r42B = r4_B / r2_B
            
                # 4. Compute C8
                
                #C8_AB[A][B] *= (3.0/2.0) * (r42A + r42B) * self.scale8
        
                # from grimme:
                qa = math.sqrt(constants.atomic_number[ele_A]) * r42A
                qb = math.sqrt(constants.atomic_number[ele_B]) * r42B
                C8_AB[A][B] *= 3*math.sqrt(qa*qb)
        
                #C8_AB[A][B] *= 1.5 * math.sqrt(r42A + r42B) * self.scale8
                #C8_AB[A][B] *= 1.5 * (r42A + r42B) * self.scale8

                # Note: The above expression was derived from Starckschall and Gordon (1972)
                #       MEDFF uses a similar expression, but with the sum of the r42 terms
                #       with in a square root, not sure why. 
        return C8_AB


    def compute_c10_coeffs(self, C6_AB, C8_AB):

        # C10 coefficients are computed with the recursion relation:
        #
        #   C10 = (49/40) * C8^2 / C6

        C10_AB = np.copy(np.square(C8_AB))
        C10_AB = np.divide(C10_AB,C6_AB)
        C10_AB = np.multiply(C10_AB, (49.0/40.0))

        return C10_AB

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
