#!/usr/bin/env python
#
# Cell class. Define cell geometry.
#
# Tristan Bereau (2016)

import numpy as np
import cliff.helpers.utils
import cliff.helpers.constants
import logging

# Set logger
logger = logging.getLogger(__name__)

class Cell():
    'Cell geometry'

    def __init__(self, cellh):
        '''Constructor initializes H matrix and its inverse.'''
        # H matrix
        self.cellh = cellh
        # Inverse of H matrix
        self.celli = np.linalg.inv(self.cellh)

    @classmethod
    def lattice_parameters(cls, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
        '''Units: distance in Angstroem, angles in degrees.
        '''
        # Convert to radians
        alpha, beta, gamma = map(np.radians, (alpha, beta, gamma))
        # H matrix
        cellh = np.zeros((3,3))
        a = np.array((a, 0, 0))
        b = b * np.array((np.cos(gamma), np.sin(gamma), 0))
        bracket = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        c = c * np.array((np.cos(beta), bracket, np.sqrt(np.sin(beta) ** 2 - bracket ** 2)))

        cellh[:, 0] = a
        cellh[:, 1] = b
        cellh[:, 2] = c

        return cls(cellh)

    def pbc_distance(self, coords1, coords2):
        s_12  = np.dot(self.celli, coords2) - np.dot(self.celli, coords1)
        s_12 -= np.rint(s_12)
        return np.dot(self.cellh, s_12)
