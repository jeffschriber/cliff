#!/usr/bin/env python
#
# Repulsion class. Compute repulsive interaction based on anisotropic Hirshfeld
# ratios.
#
# Tristan Bereau (2017)

import numpy as np
import math
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
from cliff.helpers.system import System
from cliff.helpers.cell import Cell
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.polarizability import Polarizability
from cliff.atomic_properties.atomic_density import AtomicDensity
import logging
from numba import jit

# Set logger
logger = logging.getLogger(__name__)


class Repulsion:
    '''
    Repulsion class. Compute repulsive interaction based on overlap integrals.
    '''

    def __init__(self,options, sys, cell, reps=None, v1=False):
        self.systems = [sys]
        self.atom_in_system = [0]*len(sys.elements)
        logger.setLevel(options.get_logger_level())
        # Need a unit cell for distance calculations
        self.cell = cell
        # energy
        self.energy = 0.0
        # Predict valence widths for sys
        self.adens = AtomicDensity(options)
        self.adens.load_ml()
        # self.adens.load_ml_env()
        self.adens.predict_mol(sys)
        # Combined system for combined valence-width prediction
        self.sys_comb = sys
        self.adens.predict_mol(self.sys_comb)
        # Load variables from config file
        # Atom types defined



        # Be sure ordering is canonical
        self.rep = options.get_exchange_int_params()
        if reps != None:
            ele_ad = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
            for n, ele in enumerate(ele_ad):
                self.rep[ele] = reps[n]

    def add_system(self, sys):
        self.systems.append(sys)
        last_system_id = self.atom_in_system[-1]
        self.atom_in_system += [last_system_id+1]*len(sys.elements)
        self.sys_comb = self.sys_comb + sys
        self.adens.predict_mol(sys)
        self.sys_comb.populations, self.sys_comb.valence_widths = [], []
        # Refinement
        for s in self.systems:
            # self.adens.predict_mol_env(s,self.sys_comb)
            self.sys_comb.populations    = np.append(self.sys_comb.populations,
                                                        s.populations)
            self.sys_comb.valence_widths = np.append(self.sys_comb.valence_widths,
                    s.valence_widths)
        return None

    def compute_repulsion(self, inter_type):
        'Compute repulsive interaction'
        # Setup list of atoms to sum over
        atom_coord  = [crd for sys in self.systems
                        for _,crd in enumerate(sys.coords)]
        atom_ele    = [ele for sys in self.systems
                        for _,ele in enumerate(sys.elements)]
        atom_bnd   = [bnd for sys in self.systems
                            for _,bnd in enumerate(sys.bonded_atoms)]
        atom_typ   = [typ for sys in self.systems
                            for _,typ in enumerate(sys.atom_types)]
        #print("Types", atom_typ)# for getting U
        populations = [p for _,p in enumerate(self.sys_comb.populations)]
        valwidths   = [v/constants.a2b for sys in self.systems
                        for _,v in enumerate(sys.valence_widths)]
        if inter_type == "slater_isa":
            self.energy = sum([self.slater_isa(
                    atom_ele[i], atom_coord[i], valwidths[i], atom_typ[i], atom_bnd[i],
                    atom_ele[j], atom_coord[j], valwidths[j], atom_typ[j], atom_bnd[j])
                    for i,_ in enumerate(atom_coord)
                    for j,_ in enumerate(atom_coord)
                    if self.different_mols(i,j) and i<j])
        elif inter_type == "slater_mbis":
 
            self.energy = sum([self.slater_mbis(
                    atom_coord[i], populations[i], valwidths[i], atom_typ[i],
                    atom_coord[j], populations[j], valwidths[j], atom_typ[j])
                    for i,_ in enumerate(atom_coord)
                    for j,_ in enumerate(atom_coord)
                    if self.different_mols(i,j) and i<j])
        else:
            logger.error("Interaction type not implemented")
        logger.info("Energy: %7.4f kcal/mol" % self.energy)
        return self.energy

    def different_mols(self, i, j):
        """
        Returns True if atom indices i and j belong to different systems.
        If there's only one system, don't distinguish between different molecules.
        """
        if len(self.systems) == 1:
            return True
        else:
            return self.atom_in_system[i] is not self.atom_in_system[j]

    #def slater_isa(self, ele_i, coord_i, v_i, typ_i, bnd_i,
    #        ele_j, coord_j, v_j, typ_j, bnd_j):
    #    """Compute repulsion based on Slater-ISA method
    #    (Van Vleet et al., arXiv 1606.00734)"""
    #    vec = self.cell.pbc_distance(coord_i, coord_j)
    #    rij = np.linalg.norm(vec)
    #    bij = 1./ np.sqrt(v_i * v_j)
    #    bijrij = constants.a2b * bij * rij
    #    c1i,c2i = utils.cosangle_two_atoms_inter(ele_i,coord_i,bnd_i,ele_j,coord_j,bnd_j,vec)
    #    c1j,c2j = utils.cosangle_two_atoms_inter(ele_j,coord_j,bnd_j,ele_i,coord_j,bnd_j,-vec)
    #    cthet1 = c1i*c1j
    #    cthet2 = max(c2i,c2j)
    #    bmr = (1+bijrij+1/3.*bijrij**2)*np.exp(-bijrij)
    #    # Default interaction
    #    bm  = np.pi/bij**3 * bmr
    #    # Hbond
    #    if typ_i in ["HN","HO", "HS"] and ele_j in ["O","N","S"] \
    #        or typ_j in ["HN","HO","HS"] and ele_i in ["O","N","S"]:
    #        bm = np.pi/(2*bij**4)*bmr*max(c1i,c1j)*bijrij
    #    elif typ_i in ["HN","HO","HS"] and typ_j in ["HN","HO","HS"]:
    #        bm = np.pi/(15*bij**5)*((bijrij**3+6*bijrij**2+15*bijrij+15)*cthet2
    #                                   -(bijrij**2+3*bijrij+3)*bijrij**2*cthet1) \
    #                                   *np.exp(-bijrij)
    #    return self.scale_rep[ele_i]*self.scale_rep[ele_j] * bm

    def slater_mbis_v1(self, coord_i, N_i, v_i, typ_i, coord_j, N_j, v_j, typ_j):
        "Repulsion model as described in Vandenbrande et al., JCTC, 13 (2017)"
        vec = self.cell.pbc_distance(coord_i, coord_j)
        rij = np.linalg.norm(vec)
        prefactor = 1.
        ### THIS IS THE ORIGINAL
        for typ in [typ_i, typ_j]:

            prefactor *= self.rep[typ] if typ in self.rep.keys() \
                                        else self.rep[typ[0]]
        ## END OF ORIGINAL
        #print(typ_i, typ_j, slater_mbis_funcform(rij, N_i, v_i, N_j, v_j)) # for getting U
        return prefactor * slater_mbis_funcform(rij, N_i, v_i, N_j, v_j)

    def slater_mbis(self, coord_i, N_i, v_i, typ_i, coord_j, N_j, v_j, typ_j):
        "Repulsion model as described in Vandenbrande et al., JCTC, 13 (2017)"
        vec = self.cell.pbc_distance(coord_i, coord_j)
        rij = np.linalg.norm(vec)
        prefactor = 1.
        for typ in [typ_i, typ_j]:

            prefactor *= self.rep[typ] if typ in self.rep.keys() \
                                        else self.rep[typ[0]]
        #print( typ_i, typ_j, slater_mbis_funcform(rij, N_i, v_i, N_j, v_j)) # for getting U
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
