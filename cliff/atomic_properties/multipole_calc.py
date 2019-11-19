#!/usr/bin/env python
#
# MultipoleCalc class. Compute static multipole electrostatics.
#
# Tristan Bereau (2017)

import numpy as np
from system import System
from numpy import exp
from calculator import Calculator
from atomic_density import AtomicDensity
from cell import Cell
import logging
import constants
import utils

# Set logger
logger = logging.getLogger(__name__)

class MultipoleCalc:
    'Mulitpole_calc class computes static multipole electrostatics'

    def __init__(self, sys, cell):
        self.systems = [sys]
        self.atom_in_system = [0]*len(sys.elements)
        logger.setLevel(sys.get_logger_level())
        self.cell = cell
        self.mtps_cart = None
        self.energy_static = 0.0
        self.energy_cp = 0.0
        # Core charges
        self.charge_core = []
        # Predict valence widths for sys
        self.adens = AtomicDensity(Calculator())
        self.adens.load_ml()
        self.adens.predict_mol(sys)
        # self.adens.load_ml_env()
        sys.chg_core = [mtp[0]-N for mtp,N in zip(sys.multipoles, sys.populations)]
        # Combined system for combined valence-width prediction
        self.sys_comb = sys
        self.adens.predict_mol(self.sys_comb)
        # Type of penetration correction
        self.penet_type = str(self.systems[0].Config.get(
                "chargepenetration","type")).lower()
        if self.penet_type == "wang":
            self.cp_alpha = self.systems[0].Config.getfloat(
                "chargepenetration","alpha")
            self.cp_beta = self.systems[0].Config.getfloat(
                "chargepenetration","beta")

    def add_system(self, sys):
        self.systems.append(sys)
        last_system_id = self.atom_in_system[-1]
        self.atom_in_system += [last_system_id+1]*len(sys.elements)
        self.sys_comb = self.sys_comb + sys
        self.sys_comb.populations, self.sys_comb.chg_core, \
            self.sys_comb.valence_widths = [], [], []
        self.adens.predict_mol(sys)
        # Refinement
        for s in self.systems:
            # self.adens.predict_mol_env(s,self.sys_comb)
            s.chg_core = [mtp[0]-N for mtp,N in zip(s.multipoles, s.populations)]
            self.sys_comb.populations    = np.append(self.sys_comb.populations,
                                                        s.populations)
            self.sys_comb.chg_core       = np.append(self.sys_comb.chg_core,
                    [mtp[0]-N for mtp,N in zip(s.multipoles, s.populations)])
            self.sys_comb.valence_widths = np.append(self.sys_comb.valence_widths,
                    s.valence_widths)
        return None

    def convert_mtps_to_cartesian(self, stone_convention):
        'Convert spherical MTPs to cartesian'
        num_atoms = sum(sys.num_atoms for sys in self.systems)
        # q={0,1,2} => 1+3+9 = 13 parameters
        self.mtps_cart = np.zeros((num_atoms,13))
        idx = 0
        for sys in self.systems:
            if len(sys.multipoles) == 0:
                logger.error("Multipoles not initialized for system %s!"
                    % sys)
                exit(1)
            for i in range(sys.num_atoms):
                self.mtps_cart[idx][0] = sys.multipoles[i][0]
                self.mtps_cart[idx][1] = sys.multipoles[i][1]
                self.mtps_cart[idx][2] = sys.multipoles[i][2]
                self.mtps_cart[idx][3] = sys.multipoles[i][3]
                # Convert spherical quadrupole
                cart_quad = utils.spher_to_cart(
                                sys.multipoles[i][4:9], stone_convention)
                # xx, xy, xz, yx, yy, yz, zx, zy, zz
                self.mtps_cart[idx][4:13] = cart_quad.reshape((1,9))
                idx += 1
        return None

    def mtp_energy(self, stone_convention=False):
        'Convert multipole interactions'
        self.convert_mtps_to_cartesian(stone_convention)
        # Setup list of atoms to sum over
        atom_coord = [crd for sys in self.systems
                            for _, crd in enumerate(sys.coords)]
        atom_ele   = [ele for sys in self.systems
                            for _, ele in enumerate(sys.elements)]
        valwidths = [v for _,v in enumerate(self.sys_comb.valence_widths)]
        # Loop over all pairs (au) -- hbohr2kcalmol
#        self.energy_static = constants.au2kcalmol * sum([np.dot(
#                    self.mtps_cart[i].T,
#                    np.dot(interaction_tensor(crdi, crdj, self.cell), self.mtps_cart[j]))
#                            for i, crdi in enumerate(atom_coord)
#                            for j, crdj in enumerate(atom_coord)
#                            if self.different_mols(i,j) and j>i])

        for i, crdi in enumerate(atom_coord):
            for j, crdj in enumerate(atom_coord):
                if self.different_mols(i,j) and j>i:
                    e_static = constants.au2kcalmol * np.dot(
                                self.mtps_cart[i].T,
                                np.dot(interaction_tensor(crdi, crdj, self.cell), self.mtps_cart[j]))

        #            print(e_static)
                    self.energy_static += e_static



        it = interaction_tensor(atom_coord[0], atom_coord[1], self.cell)
        # print ""
        # for i, crdi in enumerate(atom_coord):
        #     for j, crdj in enumerate(atom_coord):
        #         if self.different_mols(i,j) and j>i:
        #             print i+1,j+1,atom_ele[i],atom_ele[j], constants.au2kcalmol * np.dot(self.mtps_cart[i].T,
        #                 np.dot(interaction_tensor(crdi, crdj, self.cell), self.mtps_cart[j]))
        logger.info("Static energy: %7.4f kcal/mol" % self.energy_static)
        if self.penet_type == "vandenbrande":
            # Screened charge
            chg_scr   = [N for _,N in enumerate(self.sys_comb.populations)]
            chg_core  = [q for _,q in enumerate(self.sys_comb.chg_core)]
            self.energy_cp     = constants.au2kcalmol * sum([
                    self.penetration_vandenbrande(
                        atom_ele[i], crdi, valwidths[i], chg_core[i], chg_scr[i],
                        atom_ele[j], crdj, valwidths[j], chg_core[j], chg_scr[j])
                            for i, crdi in enumerate(atom_coord)
                            for j, crdj in enumerate(atom_coord)
                            if self.different_mols(i,j) and j>i])
        elif self.penet_type == "wang":
            chg = [mtp[0] for sys in self.systems for _,mtp in enumerate(sys.multipoles)]
            nuc = [constants.cp_Z[ele] for ele in atom_ele]
            self.energy_cp     = constants.au2kcalmol * sum([
                    self.penetration_wang(
                        crdi, valwidths[i], chg[i], nuc[i],
                        crdj, valwidths[j], chg[j], nuc[j])
                            for i, crdi in enumerate(atom_coord)
                            for j, crdj in enumerate(atom_coord)
                            if self.different_mols(i,j) and j>i])
        else:
            logger.error("Unknown penetration correction type %s!" % self.penet_type)
        logger.info("Charge penetration energy: %7.4f kcal/mol" % self.energy_cp)
        #print("Charge penetration energy: %7.4f kcal/mol" % self.energy_cp)
        return self.energy_static, self.energy_cp, it

    def different_mols(self, i, j):
        """
        Returns True if atom indices i and j belong to different systems.
        """
        return self.atom_in_system[i] is not self.atom_in_system[j]

    def charge_mult_damping(self, valwidth, rij):
        damp = (1.0 + (rij/(2.0*valwidth)))*np.exp(-rij/valwidth)
        return damp

    def mult_mult_damping(self, val_i, val_j, rij):
        
        denom = (val_i*val_i - val_j*val_j)
        if abs(denom) < 1e-6:
            print("WARNING: Denominator = ", denom)
        damp = (val_i**4/denom**2)*(1.0 + (rij/(2.0*val_i)) - (2.0*val_j**2/denom)) * np.exp(-rij/val_i)       
        return damp


    def penetration_vandenbrande(self, ele_i, crd_i, v_i, core_i, nscr_i,
                                            ele_j, crd_j, v_j, core_j, nscr_j):
        """
        Compute charge penetration between two sites.
        See: Vandenbrande et al., JCTC 13 (2017)
        """
        vec = self.cell.pbc_distance(crd_i, crd_j)
        rij = np.linalg.norm(vec) * constants.a2b
        v_i2, v_j2 = v_i**2, v_j**2
        rvi, rvj = rij/v_i, rij/v_j
        #cisj = core_i * nscr_j / rij * (1+rvj/2.) * np.exp(-rvj)
        #cjsi = core_j * nscr_i / rij * (1+rvi/2.) * np.exp(-rvi)
        cisj = core_i * nscr_j / rij * self.charge_mult_damping(v_j,rij) 
        cjsi = core_j * nscr_i / rij * self.charge_mult_damping(v_i,rij) 
        if abs(v_i-v_j) > 1e-3:
        #if abs(v_i-v_j) > 1e-3:
            #fij = v_i2**2/(v_i2-v_j2)**2 * (1 + rvi/2. - 2*v_j2/(v_i2-v_j2)) * np.exp(-rvi)
            #fji = v_j2**2/(v_j2-v_i2)**2 * (1 + rvj/2. - 2*v_i2/(v_j2-v_i2)) * np.exp(-rvj)
            fij = self.mult_mult_damping(v_i, v_j, rij) 
            fji = self.mult_mult_damping(v_j, v_i, rij)
            return cisj * cjsi - nscr_i*nscr_j / rij * (fij + fji)
        else:
            rv = rij/v_i
            rv2, rv3, rv4 = rv**2, rv**3, rv**4
            exprv = np.exp(-rv)
            f1 = 1/rij * (1+11/16.*rv+3/16.*rv2+1/48.*rv3)*exprv
            f2 = (v_j-v_i)/(96*v_i2) * (15+15*rv+6*rv2+rv3) * exprv
            f3 = (v_j-v_i)**2/(320*v_i**3) * (20+20*rv+5*rv2-5/3.*rv3-rv4) * exprv
            return cisj * cjsi - nscr_i*nscr_j * (f1+f2+f3)

    def penetration_wang(self, crd_i, v_i, q_i, Z_i, crd_j, v_j, q_j, Z_j):
        """
        Compute charge penetration between two sites.
        See: Wang et al., JCTC 11 (2017)
        """
        vec = self.cell.pbc_distance(crd_i, crd_j)
        rij = np.linalg.norm(vec) * constants.a2b
        return (Z_i * Z_j - Z_i * (Z_j - q_j) * (1 - exp(-self.cp_alpha/v_j*rij)) \
                    - Z_j * (Z_i - q_i) * (1 - exp(-self.cp_alpha/v_i*rij)) \
                    + (Z_i - q_i) * (Z_j - q_j) * (1 - exp(-self.cp_beta/v_i*rij)) \
                        * (1 - exp(-self.cp_beta/v_j*rij))
                        - q_i*q_j )/rij

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
        #15*y*z*(7*x2-3*r2)*ri9 # xxyz xxzy xyxz xyzx xzyx yxzx
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
 
