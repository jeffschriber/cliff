#! /usr/bin/env python


"""
cliff.py
Component-based Learned Intermolecular Force Field

Handles the primary functions


"""

import os


import numpy as np
import configparser
import sys
import argparse
import glob
import time
import json
import logging
import multiprocessing as mp

import cliff
from cliff.helpers.options import Options
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
import cliff.helpers.utils as Utils
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.atomic_density import AtomicDensity
from cliff.atomic_properties.multipole import Multipole
from cliff.components.electrostatics import Electrostatics
from cliff.components.repulsion import Repulsion
from cliff.components.induction_calc import InductionCalc
from cliff.components.dispersion import Dispersion

def set_nthread(nthread):
    os.environ["OMP_NUM_THREADS"]    = str(nthread) 
    os.environ["MKL_NUM_THREADS"]    = str(nthread)
    os.environ["NUMEXPR_NUM_THREADS"]= str(nthread)

def init_args():
    parser = argparse.ArgumentParser(description="CLIFF: a Component-based Learned Intermolecular Force Field")
    parser.add_argument('-i','--input', type=str, help='Location of input configuration file')

    parser.add_argument('-f','--files', type=str, help='Directory of monomer xyz files')

    parser.add_argument('-n','--name', type=str, help='Output job name')

    parser.add_argument('-r','--ref', type=str, help='xyz of reference monomer')

    parser.add_argument('-p','--nproc', type=int, help='Number of threads for numpy')

    return parser.parse_args()

def get_infile(inpt):
    
    infile = ""

    if inpt == None:
        # If no input provided, look for the default
        if os.path.exists('config.ini'):
            infile = 'config.ini'            
    else:
        if not os.path.exists(inpt):
            raise Exception("File {} does not exist!".format(inpt))
        else:
            infile = inpt                

    return infile

def load_models(options, ref=None):
    #initializes Hirshfeld class
    hirsh = Hirshfeld(options, ref) 
    adens = AtomicDensity(options, ref)

    #load KRR model for Hirshfeld as specified in the config.ini file
    hirsh.load_ml() 
    adens.load_ml()

    #load multipoles with aSLATM representation
    mtp = Multipole(options,ref) 

    # Predict the references, save to disk
    if ref is not None:
        sys = System(options,ref)
        hirsh.predict_mol(sys, force_predict = True)
        hirsh.save(sys)  

        adens.predict_mol(sys, force_predict = True)
        adens.save(sys)

        mtp.predict_mol(sys, force_predict = True)
        sys.save_mtp()

    return [hirsh, adens, mtp]

def get_energy(filenames, models, options, timer=None):
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    hirsh = models[0]
    adens = models[1]
    mtp_ml = models[2]
   
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)
    
    #loads monomer geometries
    mols = []
    xyzs = [] 
    for xyz in filenames:
        #computes descriptor for each monomer   
        mols.append(System(options, xyz))
        xyzs.append(xyz)
    
#    mtp_ml.predict_mols(mols)
    logger.info("")
    for mol,xyz in zip(mols,xyzs):
        logger.info("    Predicting atomic properties for {}".format(xyz))
        #predicts monomer multipole moments for each monomer
        mtp_ml.predict_mol(mol)
        #predicts Hirshfeld ratios using KRR
        hirsh.predict_mol(mol)
        adens.predict_mol(mol)
    
    #initializes relevant classes with monomer A
    mtp = Electrostatics(options,mols[0], cell)
    ind = InductionCalc(options, mols[0], cell)
    rep = Repulsion(options, mols[0], cell)
    disp = Dispersion(options, mols[0], cell) 
    
    #adds monomer B
    for mol in mols[1:]:
        mtp.add_system(mol)
        ind.add_system(mol)
        rep.add_system(mol)
        disp.add_system(mol)
    
    #computes electrostatic, induction and exchange energies
    t1 = time.time()
    elst = mtp.mtp_energy()
    elst_time = time.time() - t1

    t2 = time.time()
    indu = ind.polarization_energy(options)
    ind_time = time.time() - t2

    t3 = time.time()
    exch = rep.compute_repulsion()
    rep_time = time.time() - t3
    
    #use Hirshfeld ratios in the computation of dispersion energy
    t4 = time.time()
    disp_en = disp.compute_dispersion(hirsh)  
    disp_time = time.time() - t4

     
    timer['elst'] =  elst_time
    timer['exch'] =  rep_time
    timer['ind']  =  ind_time
    timer['disp'] =  disp_time

    # for printing
    ret = {'elst':elst, 'exch':exch, 'indu':indu, 'disp':disp_en, 'total':elst+exch+indu+disp_en}
    return ret

def print_banner(): 
    title = r''' 
                                                      
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
    ===================================================='''

    
    logger.info(title)

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

def print_ret(name, ret):
    '''
    Print out results from return dict
    '''
    logger.info("")
    logger.info("    Output summary (kcal/mol)")
    logger.info("    File Directory   |  Electrostatics |   Exchange   |   Induction   |   Dispersion  |   Total ")
    logger.info("    ----------------------------------------------------------------------------------------------") 

    with open(name + '.json','w') as out:
        json.dump(ret,out)

    with open(name + '.csv','w') as cout:
        cout.write("# Dimer, Electrostatics, Exchange, Induction, Dispersion, Total (kcal/mol)")
        for k,v in ret.items():
            logger.info("    %-27s%18.5f %14.5f %15.5f %15.5f %11.5f" % (k, v['elst'],v['exch'],v['indu'],v['disp'],v['total']))
            cout.write("\n%s,%9.5f,%9.5f,%9.5f,%9.5f,%9.5f" % (k, v['elst'],v['exch'],v['indu'],v['disp'],v['total']))

def print_timings(timer):
    logger.info("")
    logger.info("    Component Timings")
    logger.info("    Electrostatics:  %10.3f s" % timer['elst'])
    logger.info("    Exchange      :  %10.3f s" % timer['exch'])
    logger.info("    Induction     :  %10.3f s" % timer['ind'])
    logger.info("    Dispersion    :  %10.3f s" % timer['disp'])

def main(inpt=None, files=None, ref=None, nproc=None, name=None):
    # Do something if this file is invoked on its own

    infile = get_infile(inpt)

    #1. Initialize relevant variables
    options = Options(infile,name) 
    options.set_name(name) 
    logger.setLevel(options.logger_level)

    print_banner()

    if infile == "":
        logger.info("    Using default options")
    else:
        logger.info("    Loading options from {}".format(infile))

    if nproc is not None:
        logger.info("    Using {} threads".format(nproc))
    else:
        logger.info("    Using {} threads".format(mp.cpu_count()))

    job_list = Utils.file_finder(options.name,ref,files)

    ret = {}
    timer = {}


    models = load_models(options, ref)
    for filenames in job_list:

        if ref is not None:
            d_name =  filenames[0].split('/')[-1].split('.xyz')[0]
            d_name += filenames[1].split('/')[-1].split('.xyz')[0]
        else: 
            d_name = filenames[0].split('/')[-2]
        en = get_energy(filenames, models, options, timer)
        ret[d_name] = en

    print_ret(name, ret)
    print_timings(timer)

    return ret

if __name__ == "__main__":
    start = time.time()
    args = init_args()

    name = 'output'
    if args.name is not None:
        name = args.name

    if args.nproc is not None:
        set_nthread(args.nproc)


    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(name + '.log')
    logger.addHandler(fh)

    main(args.input, args.files, args.ref, args.nproc, name)
    end = time.time()

    logger.info("    CLIFF ran in {} s".format(end-start))

