#! /usr/bin/env/ python

# import relevant modules 
import cliff
from cliff.helpers.options import Options 
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.multipole import Multipole
from cliff.atomic_properties.atomic_density import AtomicDensity 
from cliff.components.cp_multipoles import CPMultipoleCalc
from cliff.components.induction_calc import InductionCalc
from cliff.components.repulsion import Repulsion
from cliff.components.dispersion import Dispersion

import os, sys, glob, math, pickle
import numpy as np
import glob
import json
import sys
from random import shuffle
from functools import reduce
import configparser
import operator

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

def get_energy(filename):
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    #1. Initialize relevant variables
    
    #loads parameters contained on the config.init file
    options = Options(testpath + 'ref_ml/config.ini') 
    
    # set the path for reference parameters
    options.set_hirshfeld_training( testpath + '../models/small/hirshfeld_model_0.3.pkl') 
    options.set_atomicdensity_training(testpath + '/../models/small/atomic_pop_width_0.3.pkl') 
    options.set_multipole_training(testpath + '/../models/small/mtp/') 
    
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)
    
    #initializes Hirshfeld class
    hirsh = Hirshfeld(options) 
    adens = AtomicDensity(options)
    
    #load KRR model for Hirshfeld as specified in the config.init file
    hirsh.load_ml() 
    adens.load_ml()
    
    #load multipoles with aSLATM representation
    mtp_ml  = Multipole(options, descriptor="slatm") 
    
    #loads monomer geometries
    mols = []
    xyzs = [] 
    for xyz in filename:
        #computes descriptor for each monomer   
        mols.append(System(options,xyz))
        xyzs.append(xyz)
    
    for mol,xyz in zip(mols,xyzs):
        #predicts monomer multipole moments for each monomer
        mtp_ml.predict_mol(mol)
        #predicts Hirshfeld ratios usin KRR
        hirsh.predict_mol(mol)
        adens.predict_mol(mol)
    
    #initializes relevant classes with monomer A
    mtp = CPMultipoleCalc(options,mols[0], cell)
    ind = InductionCalc(options,mols[0], cell)
    rep = Repulsion(options, mols[0], cell)
    dis = Dispersion(options, mols[0], cell)
    
    #adds monomer B
    for mol in mols[1:]:
        mtp.add_system(mol)
        ind.add_system(mol)
        rep.add_system(mol)
        dis.add_system(mol)
    
    #computes electrostatic, induction and exchange energies
    elst = mtp.mtp_energy()
    indu = ind.polarization_energy(options,0.5478502)
    exch = rep.compute_repulsion("slater_mbis")
    disp = dis.compute_dispersion(hirsh)
    
     
    # for printing
    print(elst, exch, indu, disp,  elst+exch+indu+disp)
    return elst, exch, indu, disp,  elst+exch+indu+disp
    
    
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
    
#with open(testpath + 'ref_ml/ref_ml.json','w') as f:
#    json.dump(current,f)
#exit()

refs = {}
with open(testpath + 'ref_ml/ref_ml.json','r') as f:
    refs = json.load(f)

def test_elst():
    for k,v in current.items():
        r = refs[k]
        en = v[0]
        assert abs(en - r[0]) < 1e-5

def test_exch():
    for k,v in current.items():
        r = refs[k]
        en = v[1]
        assert abs(en - r[1]) < 1e-5

def test_ind():
    for k,v in current.items():
        r = refs[k]
        en = v[2]
        assert abs(en - r[2]) < 1e-5

def test_disp():
    for k,v in current.items():
        r = refs[k]
        en = v[3]
        assert abs(en - r[3]) < 1e-5

def test_total():
    for k,v in current.items():
        r = refs[k]
        en = v[4]
        assert abs(en - r[4]) < 1e-4
