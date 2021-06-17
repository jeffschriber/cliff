#!/usr/bin/env python
import numpy as np
import math
import cliff.helpers.constants as constants
import cliff.helpers.utils as utils
from cliff.helpers.system import System
from cliff.helpers.cell import Cell
import logging

class Repulsion:
    '''
    Repulsion class. Compute repulsive interaction based on overlap integrals.
    '''

    def __init__(self,options, sys, cell, reps=None, v1=False):
        name = options.name
        # Set logger
        self.logger = options.logger

        self.systems = [sys]
        self.atom_in_system = [0]*len(sys.elements)
        self.logger.setLevel(options.logger_level)
        # Need a unit cell for distance calculations
        self.cell = cell
        # energy
        self.energy = 0.0
        self.sys_comb = sys
        # Load variables from config file

        self.rep = options.exch_int_params
        
        self.decompose = True
        self.at_exch = np.zeros((0,0))
        

    def add_system(self, sys):
        self.systems.append(sys)
        return None

    def compute_repulsion(self):
        'Compute repulsive interaction'
        # Setup list of atoms to sum over

        atom_coord = []    
        v_widths = []
        params = []
        for sys in self.systems:
            atom_coord.append([crd*constants.a2b for crd in sys.coords])
            params.append([self.rep[typ] for typ in sys.atom_types])
            v_widths.append([v for v in sys.valence_widths])
 
        nsys = len(self.systems)
        self.energy = 0.0
        for s1 in range(nsys):
            for s2 in range(s1+1, nsys):
                r = utils.build_r(atom_coord[s1], atom_coord[s2], self.cell)
                ovp = utils.slater_ovp_mat(r,v_widths[s1],v_widths[s2])
                self.energy += np.dot(params[s1], np.matmul(ovp,params[s2]))

                if self.decompose:
                    self.at_exch = np.zeros((len(atom_coord[s1]), len(atom_coord[s2])))
                    for i in range(len(atom_coord[s1])): 
                        for j in range(len(atom_coord[s2])): 
                            self.at_exch[i,j] = ovp[i,j] * params[s1][i] * params[s2][j] * constants.au2kcalmol


        self.energy *= constants.au2kcalmol
        self.logger.debug("Energy: %7.4f kcal/mol" % self.energy)
        return self.energy
