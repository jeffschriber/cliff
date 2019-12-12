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
    options = Options(testpath + 'defaults/config.ini') 
    
    # set the path for reference parameters
    options.set_hirshfeld_filepath(testpath + '/s30/hirshfeld_data/') 
    options.set_atomicdensity_refpath(testpath + '/s30/multipoles_and_valence_parameters/') 
    options.set_multipole_ref_path(testpath + '/s30/multipoles_and_valence_parameters/') 
    
    # set a bunch of options
    options.set_logger_level(50)

    options.set_hirshfeld_max_neighbors(12)
    options.set_hirshfeld_krr_kernel('laplacian')
    options.set_hirshfeld_krr_sigma(1000.0)
    options.set_hirshfeld_krr_lambda(1e-9)
    options.set_hirshfeld_file_read(True)

    options.set_atomicdensity_krr_kernel('laplacian')
    options.set_atomicdensity_krr_sigma(1000.0)
    options.set_atomicdensity_krr_lambda(1e-9)
    options.set_atomicdensity_ref_adens(True)

    options.set_multipole_kernel('laplacian')
    options.set_multipole_krr_sigma(10.0)
    options.set_multipole_krr_lambda(1e-3)
    options.set_multipole_correct_charge(True)
    options.set_multipole_ref_mtp(True)
    options.set_mtp_to_disk(False)

    induction_p = {
    'Cl1' :  2.41792428, 
    'F1'  :  2.41939871, 
    'S1'  :  2.42544107, 
    'S2'  :  0.2566964 ,
    'HS'  :  3.63958487,
    'HC'  :  4.94372272, 
    'HN'  :  3.40158855, 
    'HO'  :  2.43802344, 
    'C4'  :  1.57450169, 
    'C3'  :  1.28488505, 
    'C2'  :  0.75142714,
    'N3'  :  2.8577943 , 
    'N2'  :  0.74496709, 
    'N1'  :  2.57768268, 
    'O1'  :  2.16232598, 
    'O2'  :  0.841367}

    exch_int_params = {
    'Cl1' :  24.78385622 , 
    'F1'  :  20.96059488 , 
    'S1'  :  22.7512158  , 
    'S2'  :  26.02156528 ,
    'HS'  :  26.6079981  ,
    'HC'  :  23.12090061 , 
    'HN'  :  33.28637123 , 
    'HO'  :  26.55016745 , 
    'C4'  :  25.98532033 , 
    'C3'  :  27.79120453 , 
    'C2'  :  34.40369053 ,
    'N3'  :  22.77064348 , 
    'N2'  :  27.0469614  , 
    'N1'  :  26.87691725 , 
    'O1'  :  21.99186584 , 
    'O2'  :  22.72871279
    }
    
    elst_cp_exp = {
    'Cl1' : 3.3002,
    'F1'  : 4.4217,
    'S1'  : 3.1779,
    'S2'  : 2.6895,
    'HS'  : 3.2298,
    'HC'  : 4.1696,
    'HN'  : 2.6589, 
    'HO'  : 3.1564,
    'C4'  : 3.0832,
    'C3'  : 3.0723,
    'C2'  : 3.2478,
    'N3'  : 4.2527, 
    'N2'  : 3.7247,
    'N1'  : 3.8178,
    'O1'  : 3.7885, 
    'O2'  : 3.9713
    }

    options.set_elst_type('damped_mtp')
    options.set_damping_exponents(elst_cp_exp)

    options.set_indu_smearing_coeff(0.5478502)
    options.set_induction_omega(0.75)
    options.set_induction_conv(1e-5)
    options.set_induction_sr_params(induction_p)
    
    options.set_exchange_int_params(exch_int_params)
    
    options.set_pol_scs_cutoff(5.01451)
    options.set_disp_beta(2.40871)
    options.set_disp_radius(0.57785)
    options.set_pol_exponent(0.177346)


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
    return elst, exch, indu, disp,  elst+exch+indu+disp
    
    
filenames = {}
current = {}
mols = glob.glob(testpath + 's30/xyzs/*/*monoA-unCP.xyz')
mols = sorted(mols)
for mol in mols:
    monA = mol 
    monB = mol.strip("monoA-unCP.xyz") + "-monoB-unCP.xyz"
    filename = [monB,monA]
    monA = monA.split('/')[-1]
    monA = monA.split('-mon')[0]
    filenames[monA] = filename
    current[monA] = get_energy(filename)
    

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
