#! /usr/bin/env/ python

# import relevant modules 
import cliff
from cliff.helpers.options import Options 
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.multipole_ml_bset import MultipoleMLBSet
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
    options = Options(testpath + 's30/config.ini') 
    
    # set the path for reference parameters
    options.set_hirshfeld_filepath(testpath + '/s30/hirshfeld_data/') 
    options.set_atomicdensity_refpath(testpath + '/s30/multipoles_and_valence_parameters/') 
    options.set_multipole_ref_path(testpath + '/s30/multipoles_and_valence_parameters/') 
    
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)
    
    #initializes Hirshfeld class
    hirsh = Hirshfeld(options) 
    adens = AtomicDensity(options)
    
    #load KRR model for Hirshfeld as specified in the config.init file
    hirsh.load_ml() 
    adens.load_ml()
    
    #load multipoles with aSLATM representation
    mtp_ml  = MultipoleMLBSet(options, descriptor="slatm") 
    
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
        hirsh.predict_mol(mol,"krr")
        adens.predict_mol(mol)
    
    #initializes relevant classes with monomer A
    mtp = CPMultipoleCalc(options,mols[0], cell)
    ind = InductionCalc(options,mols[0], cell)
    rep = Repulsion(options, mols[0], cell)
    
    #adds monomer B
    for mol in mols[1:]:
        mtp.add_system(mol)
        ind.add_system(mol)
        rep.add_system(mol)
    
    #computes electrostatic, induction and exchange energies
    elst = mtp.mtp_energy()
    indu = ind.polarization_energy(options,0.5478502)
    exch = rep.compute_repulsion("slater_mbis")
    
    #creat dimer
    dimer = reduce(operator.add, mols)

    #compute hirshfeld_ratios in the dimer basis 
    hirsh.predict_mol(dimer, "krr")   
    
    #use Hirshfeld ratios in the computation of dispersion energy
    #as disp = E_dim - E_monA - E_monB
    disp = 0.0
    for i, mol in enumerate([dimer] + mols):
        fac = 1.0 if i == 0 else -1.0
        #initialize Dispersion class 
        mbd = Dispersion(options, mol, cell)
        #compute C6 coefficients
        mbd.compute_csix()
        #compute anisotropic characteristic frequencies
        mbd.compute_freq_scaled_anisotropic()
        #execute MBD protocol
        mbd.mbd_protocol(radius=0.57785, beta=2.40871,scs_cutoff=5.01451)
        disp += fac * mbd.energy
     
    # for printing
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
    
#with open(testpath + 's30/s30_ref.json','w') as f:
#    json.dump(current,f)
#exit()

refs = {}
with open(testpath + 's30/s30_ref.json','r') as f:
    refs = json.load(f)

def test_elst():
    for k,v in current.items():
        r = refs[k]
        en = v[0]
        assert (en - r[0]) < 1e-5

def test_exch():
    for k,v in current.items():
        r = refs[k]
        en = v[1]
        assert (en - r[1]) < 1e-5

def test_ind():
    for k,v in current.items():
        r = refs[k]
        en = v[2]
        assert (en - r[2]) < 1e-5

def test_disp():
    for k,v in current.items():
        r = refs[k]
        en = v[3]
        assert (en - r[3]) < 1e-5

def test_total():
    for k,v in current.items():
        r = refs[k]
        en = v[4]
        assert (en - r[4]) < 1e-5
