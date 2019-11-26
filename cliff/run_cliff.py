#! /usr/bin/env python


"""
cliff.py
Component-based Learned Intermolecular Force Field

Handles the primary functions


"""

import numpy as np
from functools import reduce
import configparser
import operator
import sys
import argparse
import os
import glob

import cliff
from cliff.helpers.options import Options
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
import cliff.helpers.utils as Utils
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.multipole_ml_bset import MultipoleMLBSet
from cliff.components.cp_multipoles import CPMultipoleCalc
from cliff.components.repulsion import Repulsion
from cliff.components.induction_calc import InductionCalc
from cliff.components.dispersion import Dispersion

def init_args():
    parser = argparse.ArgumentParser(description="CLIFF: a Component-based Learned Intermolecular Force Field")
    parser.add_argument('-i','--input', type=str, help='Location of input configuration file')

    parser.add_argument('-f','--files', type=str, help='Directory of monomer xyz files')

    return parser.parse_args()

def get_infile(inpt):
    
    infile = ""

    if inpt == None:
        # If no input provided, look for the default
        if os.path.exists('config.ini'):
            infile = 'config.ini'            
        else:
            raise Exception("Cannot find input file")
    else:
        if not os.path.exists(inpt):
            raise Exception("File {} does not exist!".format(inpt))
        else:
            infile = inpt                

    print("    Loading options from {}".format(infile))
    return infile

def get_energy(filenames, config):
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    #1. Initialize relevant variables
    options = Options(config) 
    
   
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)
    
    #initializes Hirshfeld class
    hirsh = Hirshfeld(options) 
    
    #load KRR model for Hirshfeld as specified in the config.init file
    hirsh.load_ml() 
    
    #load multipoles with aSLATM representation
    mtp_ml  = MultipoleMLBSet(options, descriptor="slatm") 
    
    #loads monomer geometries
    mols = []
    xyzs = [] 
    for xyz in filenames:
        #computes descriptor for each monomer   
        mols.append(System(options, xyz))
        xyzs.append(xyz)
    
    for mol,xyz in zip(mols,xyzs):
        #predicts monomer multipole moments for each monomer
        mtp_ml.predict_mol(mol)
        #predicts Hirshfeld ratios usin KRR
        hirsh.predict_mol(mol,"krr")
    
    #initializes relevant classes with monomer A
    mtp = CPMultipoleCalc(options,mols[0], cell)
    ind = InductionCalc(options, mols[0], cell)
    rep = Repulsion(options, mols[0], cell)
    
    #adds monomer B
    for mol in mols[1:]:
        mtp.add_system(mol)
        ind.add_system(mol)
        rep.add_system(mol)
    
    #computes electrostatic, induction and exchange energies
    elst = mtp.mtp_energy()
    indu = ind.polarization_energy(options)
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
        mbd.mbd_protocol(None,None,None)
        disp += fac * mbd.energy
     
    # for printing
    return elst, exch, indu, disp,  elst+exch+indu+disp

def print_banner(): 
    title = ''' 
                                                      
                                      ___               
                                     /   \              
                                  __/     \                 
                               __/         \__
                     _        /               \         
                   _/ \   ___/                 \        
                __/    \_/                      \   
               / _____ _      _____ _____ _____  |  
              / / ____| |    |_   _|  ___|  ___| |  
             / | |    | |      | | | |__ | |__   |  
            /  | |    | |      | | |  __||  __|  |   
           /   | |____| |____ _| |_| |   | |     |   
          /     \_____|______|_____|_|   |_|     |  

    ====================================================
    A Component-based Learned Intermolecular Force Field

    Jeffrey B. Schriber, C. David Sherrill (2019)
    ====================================================
'''

    
    print(title)

def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "    The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t    - Adapted from Henry David Thoreau"
    return quote

def print_ret(ret):
    '''
    Print out results from return dict
    '''
    print("")
    print("    Output summary (kcal/mol)")
    print("    File Directory   |  Electrostatics |   Exchange   |   Induction   |   Dispersion  |   Total ")
    print("    ----------------------------------------------------------------------------------------------") 
    for k,v in ret.items():
        print("    %-17s%18.5f %14.5f %15.5f %15.5f %11.5f" % (k, v[0],v[1],v[2],v[3],v[4]))


def main(inpt=None, files=None):
    # Do something if this file is invoked on its own
    print_banner()

    infile = get_infile(inpt)
    job_list = Utils.file_finder(files)

    ret = {}

    for filenames in job_list:
        dirname = filenames[0].split('/')[-2]
        en = get_energy(filenames, infile)
        ret[dirname] = en

    print_ret(ret)

    return ret

if __name__ == "__main__":
    args = init_args()
    main(args.input, args.files)
