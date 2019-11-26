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

# Set logger
logger = logging.getLogger(__name__)


class Repulsion:
    '''
    Repulsion class. Compute repulsive interaction based on overlap integrals.
    '''

    def __init__(self,options, sys, cell, reps=None, v1=False):
        self.systems = [sys]
        self.atom_in_system = [0]*len(sys.elements)
        logger.setLevel(options.logger_level)
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

        self.rep = options.exch_int_params

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
       # atom_ele    = [ele for sys in self.systems
       #                 for _,ele in enumerate(sys.elements)]
       # atom_bnd   = [bnd for sys in self.systems
       #                     for _,bnd in enumerate(sys.bonded_atoms)]
        atom_type  = [typ for sys in self.systems
                            for _,typ in enumerate(sys.atom_types)]
        #print("Types", atom_typ)# for getting U
        populations = [p for _,p in enumerate(self.sys_comb.populations)]
        valwidths   = [v/constants.a2b for sys in self.systems
                        for _,v in enumerate(sys.valence_widths)]
 
        #self.energy = sum([utils.slater_mbis(self.cell,
        #        atom_coord[i], populations[i], valwidths[i], self.rep[atom_typ[i]],
        #        atom_coord[j], populations[j], valwidths[j], self.rep[atom_typ[j]])
        #        for i,_ in enumerate(atom_coord)
        #        for j,_ in enumerate(atom_coord)
        #        if self.different_mols(i,j) and i<j])
        self.energy = 0.0
        for i,_ in enumerate(atom_coord):
            for j,_ in enumerate(atom_coord):
                if self.different_mols(i,j) and i<j :
                    self.energy += utils.slater_mbis(self.cell, 
                    atom_coord[i], populations[i], valwidths[i], self.rep[atom_type[i]],
                    atom_coord[j], populations[j], valwidths[j], self.rep[atom_type[j]])


#                    print(atom_type[i],i,atom_type[j],j, utils.slater_mbis(self.cell,
#                        atom_coord[i], populations[i], valwidths[i], self.rep[atom_type[i]],
#                        atom_coord[j], populations[j], valwidths[j], self.rep[atom_type[j]])) 


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
