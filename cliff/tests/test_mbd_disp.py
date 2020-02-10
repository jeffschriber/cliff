#! /usr/bin/env/ python

# import relevant modules 
import cliff
from cliff.helpers.options import Options 
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.atomic_density import AtomicDensity 
from cliff.components.dispersion import Dispersion

import os, sys, glob, math, pickle
import numpy as np
import glob
import json
import sys
import operator
import pytest

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

def get_energy(filename):
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    #1. Initialize relevant variables
    
    #loads parameters contained on the config.init file
    options = Options(testpath + 'disp/config.ini') 
    
    # set the path for reference parameters
    options.set_hirshfeld_filepath(testpath + '/s30/hirshfeld_data/') 
    options.set_atomicdensity_refpath(testpath + '/s30/multipoles_and_valence_parameters/') 
    
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)
    
    #initializes Hirshfeld class
    hirsh = Hirshfeld(options) 
    adens = AtomicDensity(options)
    
    #load KRR model for Hirshfeld as specified in the config.init file
    hirsh.load_ml() 
    adens.load_ml()
    
    #loads monomer geometries
    mols = []
    xyzs = [] 
    for xyz in filename:
        #computes descriptor for each monomer   
        mols.append(System(options,xyz))
        xyzs.append(xyz)
    
    for mol,xyz in zip(mols,xyzs):
        #predicts Hirshfeld ratios usin KRR
        hirsh.predict_mol(mol)
        adens.predict_mol(mol)
    
    #initializes relevant classes with monomer A
    dis = Dispersion(options, mols[0],cell)
    
    #adds monomer B
    for mol in mols[1:]:
        dis.add_system(mol)
    
    disp = dis.compute_dispersion(hirsh)
    
    return disp
    
    
def test_disp():
    filenames = {}
    current = {}
    mols = glob.glob(testpath + 's30/xyzs/*/*monoA-unCP.xyz')
    mols = sorted(mols)
    for mol in mols:
        monA = mol 
        monB = mol.strip("monoA-unCP.xyz") + "-monoB-unCP.xyz"
        filename = [monA,monB]
        monA = monA.split('/')[-1]
        monA = monA.split('-mon')[0]
        filenames[monA] = filename
        current[monA] = get_energy(filename)
        
    
    refs = {}
    with open(testpath + 's30/s30_ref.json','r') as f:
        refs = json.load(f)

    for k,v in current.items():
        r = refs[k]
        en = v
        print(en, r[3])
        assert abs(en - r[3]) < 1e-5

