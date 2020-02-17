#!/usr/bin/env python
#
# Compute damped electrostatics
#


import numpy as np
from numpy import exp
import logging
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
from numba import jit

# Set logger
logger = logging.getLogger(__name__)
fh = logging.FileHandler('output.log')
logger.addHandler(fh)

class Electrostatics:
    'Electrostatic class computes cp-corrected multipole electrostatics'

    def __init__(self, options,sys, cell):
        self.systems = [sys]
        self.atom_in_system = [0]*len(sys.elements)
        logger.setLevel(options.logger_level)
        self.cell = cell
        self.mtps_cart = []
        self.mtps_cart_elec = []
        self.energy_elst = 0.0

        # Combined system for combined valence-width prediction
        self.sys_comb = sys
        self.exp = options.elst_damping_exponents

    def add_system(self, sys):
        self.systems.append(sys)
        last_system_id = self.atom_in_system[-1]
        self.atom_in_system += [last_system_id+1]*len(sys.elements)
        self.sys_comb = self.sys_comb + sys
        self.sys_comb.populations, self.sys_comb.valence_widths = [], []
        # Refinement
        for s in self.systems:
            self.sys_comb.valence_widths = np.append(self.sys_comb.valence_widths,
                    s.valence_widths)

        return None

    def convert_mtps_to_cartesian(self, stone_convention):
        'Convert spherical MTPs to cartesian'
        num_atoms = [sys.num_atoms for sys in self.systems]
        atom_ele = []

        # q={0,1,2} => 1+3+9 = 13 parameters
        for sys in range(len(self.systems)):
            self.mtps_cart.append(np.zeros((num_atoms[sys],13)))
       #     self.mtps_cart_elec.append(np.zeros((num_atoms[sys],13))
            atom_ele.append([ele for ele in self.systems[sys].elements])
        for s1,sys in enumerate(self.systems):
            if len(sys.multipoles) == 0:
                logger.error("Multipoles not initialized for system %s!"
                    % sys)
                exit(1)

            for i in range(sys.num_atoms):
                self.mtps_cart[s1][i][0] = sys.multipoles[i][0] - constants.atomic_number[atom_ele[s1][i]] 
                self.mtps_cart[s1][i][1] = sys.multipoles[i][1]
                self.mtps_cart[s1][i][2] = sys.multipoles[i][2]
                self.mtps_cart[s1][i][3] = sys.multipoles[i][3]
                # Convert spherical quadrupole
                cart_quad = utils.spher_to_cart(
                                sys.multipoles[i][4:9], stone_convention)
                # xx, xy, xz, yx, yy, yz, zx, zy, zz
                self.mtps_cart[s1][i][4:13] = cart_quad.reshape((1,9))

       # self.mtps_cart_elec[:] = self.mtps_cart

       # for n, atom in enumerate(atom_ele):
       #     self.mtps_cart_elec[n][0] -= constants.atomic_number[atom]

        return None

    def mtp_energy(self, stone_convention=False):
        'Convert multipole interactions'

        nsys = len(self.systems)
        self.convert_mtps_to_cartesian(stone_convention)
        # Setup list of atoms to sum over
        atom_coord = []
        atom_ele = []
        atom_nums = []
        alphas = []

        for sys in self.systems:
            atom_coord.append([crd for crd in sys.coords])
            atom_ele.append([ele for ele in sys.elements])
            atom_nums.append([constants.atomic_number[ele] for ele in sys.elements])
            alphas.append([self.exp[ele]*constants.b2a for ele in sys.atom_types])

        elst = 0.0
        elst1 = 0.0
        elst2 = 0.0
        elst3 = 0.0
        # Loop over unique interactions
        for s1 in range(nsys):
            # this is a matrix, natom x 13
            # contains ALL multipoles for sys 1
            mi = self.mtps_cart[s1]
            for s2 in range(s1+1, nsys):
                mj = self.mtps_cart[s2]

                # 1. nuclear-nuclear int
                elst0 = nuclear_rep(atom_coord[s1], atom_coord[s2], atom_nums[s1], atom_nums[s2], self.cell) 

                # 2. nuclear-MTP interaction
                ## TODO: avoid this loop over atoms in sys
                for ele, Z in enumerate(atom_nums[s1]):
                    zm_int = charge_mtp_damped_interaction(atom_coord[s1][ele], atom_coord[s2], alphas[s2], self.cell)
                    i1 = Z*np.einsum('ij,ij->i', zm_int,mj)
                    elst1 += np.sum(i1)

                for ele, Z in enumerate(atom_nums[s2]):
                    zm_int = charge_mtp_damped_interaction(atom_coord[s2][ele], atom_coord[s1], alphas[s1], self.cell)
                    i1 = Z*np.einsum('ij,ij->i', zm_int,mi)
                    elst2 += np.sum(i1)

                # 3. MTP-MTP
                for atom1 in range(len(atom_nums[s1])):
                    crdi = atom_coord[s1][atom1]        
                    alpha1 = alphas[s1][atom1]
                    mi1 = mi[atom1,:] 
                    for atom2 in range(len(atom_nums[s2])):
                        crdj = atom_coord[s2][atom2]        

                        alpha2 = alphas[s2][atom2]

                        mj1 = mj[atom2,:] 

                        d_int = full_damped_interaction(crdi, crdj, alpha1, alpha2, self.cell)

                        elst3 += np.dot(mi1.T, np.dot(d_int, mj1)) 

        elst += (elst0 + elst1 + elst2 + elst3)
                    

        self.energy_elst = elst * constants.au2kcalmol
        return self.energy_elst

def nuclear_rep(coord1, coord2, ele1, ele2, cell):

    r = constants.a2b * utils.build_r(coord1,coord2,cell)
    r = 1.0 / r
    return np.dot(ele1, np.matmul(r,ele2))

def full_damped_interaction(coord1, coord2, alpha1, alpha2, cell):
    """Full damped interaction tensor"""
    vec = constants.a2b*(cell.pbc_distance(coord1, coord2))
    r = np.linalg.norm(vec)
    r2 = r**2
    r3 = r2*r
    r4 = r2**2
    r5 = r4*r
    ri = 1./r
    ri2 = ri**2
    ri3 = ri**3
    ri5 = ri**5
    ri7 = ri**7
    ri9 = ri**9
    x = vec[0]
    y = vec[1]
    z = vec[2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    x3 = x2*x 
    y3 = y2*y
    z3 = z2*z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2
    
    a1_2 = alpha1*alpha1
    a2_2 = alpha2*alpha2
    a1_3 = a1_2*alpha1
    a2_3 = a2_2*alpha2
    a1_4 = a1_3*alpha1
    a2_4 = a2_3*alpha2

    e1r = np.exp(-1.0 * alpha1 * r)
    e2r = np.exp(-1.0 * alpha2 * r)

    it = np.zeros((13,13))

    # Get the lambdas
    lam1 = 1.0
    lam3 = 1.0
    lam5 = 1.0
    lam7 = 1.0
    lam9 = 1.0

    if abs(alpha1 - alpha2) > 1e-6:
        A = a2_2 / (a2_2 - a1_2)
        B = a1_2 / (a1_2 - a2_2)

        
        lam1 -= A*e1r
        lam1 -= B*e2r

        lam3 -= (1.0 + alpha1*r)*A*e1r 
        lam3 -= (1.0 + alpha2*r)*B*e2r

        lam5 -= (1.0 + alpha1*r + (1.0/3.0)*a1_2*r2)*A*e1r
        lam5 -= (1.0 + alpha2*r + (1.0/3.0)*a2_2*r2)*B*e2r

        lam7 -= (1.0 + alpha1*r + (2.0/5.0)*a1_2*r2 + (1.0/15.0)*a1_3*r3)*A*e1r
        lam7 -= (1.0 + alpha2*r + (2.0/5.0)*a2_2*r2 + (1.0/15.0)*a2_3*r3)*B*e2r

        lam9 -= (1.0 + alpha1*r + (3.0/7.0)*a1_2*r2 + (2.0/21.0)*a1_3*r3 + (1.0/105.0)*a1_4*r4)*A*e1r
        lam9 -= (1.0 + alpha2*r + (3.0/7.0)*a2_2*r2 + (2.0/21.0)*a2_3*r3 + (1.0/105.0)*a2_4*r4)*B*e2r

    else:
        # assume alpha1 == alpha2
    
        lam1 -= (1.0 + 0.5*alpha1*r)*e1r
        lam3 -= (1.0 + alpha1*r + 0.5*a1_2*r2)*e1r
        lam5 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3)*e1r
        lam7 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (1.0/30.0)*a1_4*r4)*e1r
        lam9 -= (1.0 + alpha1*r + 0.5*a1_2*r2 + (1.0/6.0)*a1_3*r3 + (4.0/105.0)*a1_4*r4 + (1.0/210.0)*a1_4*alpha1*r5)*e1r

    # Indices for MTP moments:
    # 00  01  02  03  04  05  06  07  08  09  10  11  12
    #  .,  x,  y,  z, xx, xy, xz, yx, yy, yz, zx, zy, zz

    # charge-charge
    it[0][0] = ri * lam1

    # charge-dipole
    for a in [1,2,3]: # xyz
        it[0,a] = -1.0 * vec[a-1] * ri3 * lam3     
    #    it[a,0] = -1.0 * vec[a-1] * ri3 * lam3     
    # charge-quadripole
    it[0,4] = lam5*3.0*x2*ri5 - lam3*ri3  # xx
    it[0,5] = lam5*3.0*x*y*ri5             # xy
    it[0,6] = lam5*3.0*x*z*ri5             # xz
    it[0,7] = it[0,5]                       # yx
    it[0,8] = lam5*3.0*y2*ri5 - lam3*ri3  # yy
    it[0,9] = lam5*3.0*y*z*ri5             # yz
    it[0,10] = it[0,6]                      # zx
    it[0,11] = it[0,9]                      # zy
    it[0,12] = lam5*3.0*z2*ri5 - lam3*ri3 # zz

    # dipole-dipole
    it[1,1] = -it[0,4] # xx
    it[1,2] = -it[0,5] # xy
    it[1,3] = -it[0,6] # xz
    it[2,2] = -it[0,8] # yy
    it[2,3] = -it[0,9] # yz
    it[3,3] = -it[1,1] -it[2,2] # zz
    # Dipole quadrupole
    it[1,4] = 15.0*x3*ri7*lam7 - 9*x*ri5*lam5  # xxx 
    it[1,5] = it[1, 7] = it[2,4] = 15*x2*y*ri7*lam7 - 3*y*ri5*lam5 # xxy xyx yxx
    it[1,6] = it[1,10] = it[3,4] = 15*x2*z*ri7*lam7 - 3*z*ri5*lam5  # xxz xzx zxx
    it[1,8] = it[2,5] = it[2,7] = 15*x*y2*ri7*lam7 - 3*x*ri5*lam5 # xyy yxy yyx
    it[1,9] = it[1,11] = it[2,6] = it[2,10] = it[3,5] = \
        it[3,7] = 15*x*y*z*ri7*lam7 # xyz xzy yxz yzx zxy zyx
    it[1,12] = it[3,6] = it[3,10] = -it[1,4] -it[1,8] # xzz zxz zzx
    it[2,8] = 15*y3*ri7*lam7 - 9*y*ri5*lam5  # yyy
    it[2,9] = it[2,11] = it[3,8] = 15*y2*z*ri7*lam7 - 3*z*ri5*lam5 # yyz yzy zyy
    it[2,12] = it[3,9] = it[3,11] = -it[1,5] -it[2,8] # yzz zyz zzy
    it[3,12] = -it[1,6] -it[2,9] # zzz
    # Quadrupole quadrupole
    it[4,4] = 105*x4*ri9*lam9 - 90*x2*ri7*lam7 + 9*ri5*lam5 # xxxx
    it[4,5] = it[4,7] =  105*x3*y*ri9*lam9 - 45*x*y*ri7*lam7 # xxxy xxyx
    it[4,6] = it[4,10] =  105*x3*z*ri9*lam9 - 45*x*z*ri7*lam7 # xxxz xxzx
    it[4,8] = it[5,5] = it[5,7] = it[7,7] = \
        105*x2*y2*ri9*lam9 - (15*x2 + 15*y2)*ri7*lam7 + 3*ri5*lam5 # xxyy xyxy xyyx yxyx
    it[4,9] = it[4,11] = it[5,6] = it[5,10] = it[6,7] = it[7,10] = \
        105*x2*y*z*ri9*lam9 - 15*z*y*ri7*lam7 # xxyz xxzy xyxz xyzx xzyx yxzx
    #    105*x2*y*z*ri9*lam9 - 45*z*y*ri7*lam7 # xxyz xxzy xyxz xyzx xzyx yxzx match v1
    # The above line is used in IPMLv1, I think it is a bug...
    it[4,12] = it[6,6] = it[6,10] = it[10,10] = \
        -it[4,4] -it[4,8] # xxzz xzxz xzzx zxzx
    it[5,8] = it[7,8] = 105*y3*x*ri9*lam9 - 45*x*y*ri7*lam7 # xyyy yxyy
    it[5,9] = it[5,11] = it[6,8] = it[7,9] = it[7,11] = it[8,10] = \
        105*y2*x*z*ri9*lam9 - 15*x*z*ri7*lam7 # xyyz xyzy xzyy yxyz yxzy yyzx
        #15*x*z*(7*y2-r2)*ri9 # xyyz xyzy xzyy yxyz yxzy yyzx
    it[5,12] = it[6,9] = it[6,11] = it[7,12] = it[9,10] = it[10,11] = \
        -it[4,5] -it[5,8] # xyzz xzyz xzzy yxzz yzzx zxzy
    it[6,12] = it[10,12] = -it[4,6] -it[5,9] # xzzz zxzz
    #it[8,8] = (105*y4-90*y2*r2+9*r4)*ri9 # yyyy
    it[8,8] = 105*y4*ri9*lam9 - 90*y2*ri7*lam7 + 9*ri5*lam5 # yyyy
    #it[8,9] = it[8,11] = 15*y*z*(7*y2-3*r2)*ri9 # yyyz yyzy
    it[8,9] = it[8,11] = 105*y3*z*ri9*lam9 - 45*y*z*ri7*lam7  # yyyz yyzy
    it[8,12] = it[9,9] = it[9,11] = it[11,11] = \
        -it[4,8] -it[8,8] # yyzz yzyz yzzy zyzy
    it[9,12] = it[11,12] = -it[4,9] -it[8,9] # yzzz zyzz
    #it[12,12] = -it[4,12] -it[8,12] # zzzz
    it[12,12] = 105*z4*ri9*lam9 - 90*z2*ri7*lam7 + 9*ri5*lam5 # zzzz
    # Symmetrize
    it = it + it.T - np.diag(it.diagonal())
    # Some coefficients need to be multiplied by -1
    for i in range(1,4):
        for j in range(0,1):
            it[i,j] *= -1.
    for i in range(4,13):
        for j in range(1,4):
            it[i,j] *= -1.

    return it 

def charge_mtp_damped_interaction(coord1, coord2, alpha2, cell):
    """Return interaction vector for charge-mtp (up to quadripoles) with damping"""
    """First pass will be damping from Rackers PCCP 2017"""
    
    # Gets rs between 1 and all in 2

    r = np.zeros(len(coord2))
    xyz = np.zeros((len(coord2),3))
    for n, coord in enumerate(coord2):
        vec = constants.a2b*(cell.pbc_distance(coord1, coord))
        xyz[n] = np.asarray(vec)
        r[n] = np.linalg.norm(vec)

    # Some intermediates
    r2 = r**2
    ri = 1./r
    ri2 = ri**2
    ri3 = ri**3
    ri5 = ri**5
    x = xyz[::,0]
    y = xyz[::,1]
    z = xyz[::,2]
    x2 = x**2
    y2 = y**2
    z2 = z**2

    lam_1 = 1.0 - np.exp(-1.0 * np.multiply(alpha2,r))
    lam_3 = 1.0 - (1.0 + np.multiply(alpha2,r)) * np.exp(-1.0*np.multiply(alpha2,r)) 
    lam_5 = 1.0 - (1.0 + np.multiply(alpha2,r) + (1.0/3.0)*np.multiply(np.square(alpha2),r2)) * np.exp(-1.0*np.multiply(alpha2,r))

    it = np.zeros((len(coord2),(13)))
    # Charge charge
    it[:,0] = ri*lam_1
    # Charge dipole
    it[:,1] = -x*ri3 * lam_3 
    it[:,2] = -y*ri3 * lam_3
    it[:,3] = -z*ri3 * lam_3
    # Charge quadrupole
    it[:,4] = 3*x2*ri5*lam_5 - ri3*lam_3   # xx
    it[:,5] = 3*x*y*ri5*lam_5  # xy
    it[:,6] = 3*x*z*ri5*lam_5  # xz
    it[:,7] = it[:,5]  # yx
    it[:,8] = 3*y2*ri5*lam_5 - ri3*lam_3 # yy
    it[:,9] = 3*y*z*ri5*lam_5  # yz
    it[:,10] = it[:,6]   # zx
    it[:,11] = it[:,9]   # zy
    it[:,12] = 3*z2*ri5*lam_5 - ri3*lam_3  # zz

   # print(it)
    return it

def interaction_tensor(coord1, coord2, cell):
    """Return interaction tensor up to quadrupoles between two atom coordinates"""
    # Indices for MTP moments:
    # 00  01  02  03  04  05  06  07  08  09  10  11  12
    #  .,  x,  y,  z, xx, xy, xz, yx, yy, yz, zx, zy, zz
    vec = constants.a2b*(cell.pbc_distance(coord1, coord2))
    r = np.linalg.norm(vec)
    r2 = r**2
    r4 = r2**2
    ri = 1./r
    ri2 = ri**2
    ri3 = ri**3
    ri5 = ri**5
    ri7 = ri**7
    ri9 = ri**9
    x = vec[0]
    y = vec[1]
    z = vec[2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2
    it = np.zeros((13,13))
    # Charge charge
    it[0,0] = ri
    # Charge dipole
    it[0,1] = -x*ri3
    it[0,2] = -y*ri3
    it[0,3] = -z*ri3
    # Charge quadrupole
    it[0,4] = (3*x2-r2)*ri5 # xx
    it[0,5] = 3*x*y*ri5 # xy
    it[0,6] = 3*x*z*ri5 # xz
    it[0,7] = it[0,5] # yx
    it[0,8] = (3*y2-r2)*ri5 # yy
    it[0,9] = 3*y*z*ri5 # yz
    it[0,10] = it[0,6] # zx
    it[0,11] = it[0,9] # zy
    it[0,12] = -it[0,4] -it[0,8] # zz
    # Dipole dipole
    it[1,1] = -it[0,4] # xx
    it[1,2] = -it[0,5] # xy
    it[1,3] = -it[0,6] # xz
    it[2,2] = -it[0,8] # yy
    it[2,3] = -it[0,9] # yz
    it[3,3] = -it[1,1] -it[2,2] # zz
    # Dipole quadrupole
    it[1,4] = -3*x*(3*r2-5*x2)*ri7 # xxx
    it[1,5] = it[1, 7] = it[2,4] = -3*y*(r2-5*x2)*ri7 # xxy xyx yxx
    it[1,6] = it[1,10] = it[3,4] = -3*z*(r2-5*x2)*ri7 # xxz xzx zxx
    it[1,8] = it[2,5] = it[2,7] = -3*x*(r2-5*y2)*ri7 # xyy yxy yyx
    it[1,9] = it[1,11] = it[2,6] = it[2,10] = it[3,5] = \
        it[3,7] = 15*x*y*z*ri7 # xyz xzy yxz yzx zxy zyx
    it[1,12] = it[3,6] = it[3,10] = -it[1,4] -it[1,8] # xzz zxz zzx
    it[2,8] = -3*y*(3*r2-5*y2)*ri7 # yyy
    it[2,9] = it[2,11] = it[3,8] = -3*z*(r2-5*y2)*ri7 # yyz yzy zyy
    it[2,12] = it[3,9] = it[3,11] = -it[1,5] -it[2,8] # yzz zyz zzy
    it[3,12] = -it[1,6] -it[2,9] # zzz
    # Quadrupole quadrupole
    it[4,4] = (105*x4-90*x2*r2+9*r4)*ri9 # xxxx
    it[4,5] = it[4,7] =  15*x*y*(7*x2-3*r2)*ri9 # xxxy xxyx
    it[4,6] = it[4,10] = 15*x*z*(7*x2-3*r2)*ri9 # xxxz xxzx
    it[4,8] = it[5,5] = it[5,7] = it[7,7] = \
        (105*x2*y2+15*z2*r2-12*r4)*ri9 # xxyy xyxy xyyx yxyx
    it[4,9] = it[4,11] = it[5,6] = it[5,10] = it[6,7] = it[7,10] = \
        15*y*z*(7*x2-1*r2)*ri9 # xxyz xxzy xyxz xyzx xzyx yxzx
    it[4,12] = it[6,6] = it[6,10] = it[10,10] = \
        -it[4,4] -it[4,8] # xxzz xzxz xzzx zxzx
    it[5,8] = it[7,8] = 15*x*y*(7*y2-3*r2)*ri9 # xyyy yxyy
    it[5,9] = it[5,11] = it[6,8] = it[7,9] = it[7,11] = it[8,10] = \
        15*x*z*(7*y2-r2)*ri9 # xyyz xyzy xzyy yxyz yxzy yyzx
    it[5,12] = it[6,9] = it[6,11] = it[7,12] = it[9,10] = it[10,11] = \
        -it[4,5] -it[5,8] # xyzz xzyz xzzy yxzz yzzx zxzy
    it[6,12] = it[10,12] = -it[4,6] -it[5,9] # xzzz zxzz
    it[8,8] = (105*y4-90*y2*r2+9*r4)*ri9 # yyyy
    it[8,9] = it[8,11] = 15*y*z*(7*y2-3*r2)*ri9 # yyyz yyzy
    it[8,12] = it[9,9] = it[9,11] = it[11,11] = \
        -it[4,8] -it[8,8] # yyzz yzyz yzzy zyzy
    it[9,12] = it[11,12] = -it[4,9] -it[8,9] # yzzz zyzz
    it[12,12] = -it[4,12] -it[8,12] # zzzz
    # Symmetrize
    it = it + it.T - np.diag(it.diagonal())
    # Some coefficients need to be multiplied by -1
    for i in range(1,4):
        for j in range(0,1):
            it[i,j] *= -1.
    for i in range(4,13):
        for j in range(1,4):
            it[i,j] *= -1.
    return it
 
