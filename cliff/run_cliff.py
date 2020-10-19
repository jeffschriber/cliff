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

    parser.add_argument('-fr','--frag',type=bool, nargs="?", default=False, help='Do fragmentation analysis')

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

    tls = time.time()
    #load KRR model for Hirshfeld as specified in the config.ini file
    hirsh.load_ml() 
    adens.load_ml()

    #load multipoles with aSLATM representation
    mtp = Multipole(options,ref) 
    tlf = time.time()
    logger.info("    ~Time spent loading ML models: {} s".format(tlf-tls))

    # Predict the references, save to disk
    if ref is not None:
        ts = time.time()
        
        pref = "/"
        for d in os.path.abspath(ref).split("/")[:-1]:
            pref += d + "/"

        # Set the path where the references are stored
        hirsh.set_save_path(pref)
        adens.set_save_path(pref)
        mtp.set_ref_path(pref)
        sys = System(options,ref)
        sys.set_mtp_save_path(pref)

        # Don't recompute references if we've already done it
        if os.path.isfile(os.path.abspath(pref) + "/" + ref.split('/')[-1].strip('.xyz') + '-h.txt'):
            logger.info("    Loading reference Hirshfeld from: {}".format(pref))
        else:
            hirsh.predict_mol(sys, force_predict = True)
            hirsh.save(sys)  

        if os.path.isfile(os.path.abspath(pref) + "/" + ref.split('/')[-1].strip('.xyz') + '-atmdns.txt'):
            logger.info("    Loading reference atomic widths from: {}".format(pref))
        else:
            adens.predict_mol(sys, force_predict = True)
            adens.save(sys)
        if os.path.isfile(os.path.abspath(pref) + "/" + ref.split('/')[-1].strip('.xyz') + '-mtp.txt'):
            logger.info("    Loading reference multipoles from: {}".format(pref))
        else:
            mtp.predict_mol(sys, force_predict = True)
            sys.save_mtp()

        tf = time.time()
        logger.info("    ~Time spent predicting atomic properties of reference: {} s".format(tf-ts))

    return [hirsh, adens, mtp]

def get_energy(filenames, models, options, name, timer=None, frag=None):
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
    
    logger.info("")
    tms = time.time()
    for mol,xyz in zip(mols,xyzs):
        logger.info("    Predicting atomic properties for {}".format(xyz))
        #predicts monomer multipole moments for each monomer
        mtp_ml.predict_mol(mol)
        #predicts Hirshfeld ratios using KRR
        hirsh.predict_mol(mol)
        adens.predict_mol(mol)
    tmf = time.time()    
    logger.info("    ~Time spent predicting atomic properties {} s".format(tmf-tms))

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

    # Save energy partitions
    #print(mtp.at_elst, np.sum(mtp.at_elst))
    #print(rep.at_exch, np.sum(rep.at_exch))
    #print(ind.at_ind, np.sum(ind.at_ind))
    #print(disp.at_disp, np.sum(disp.at_disp))

    mona = filenames[0].split('/')[-1].split('.xyz')[0]
    monb = filenames[1].split('/')[-1].split('.xyz')[0]
    natom_a = mols[0].num_atoms
    natom_b = mols[1].num_atoms

    with open(mona + '_' + monb + "_atomic.txt","w") as outf:
        for a in range(natom_a):
            for b in range(natom_b):
                elst_e =  mtp.at_elst[a,b]
                exch_e =  rep.at_exch[a,b]
                indu_e =   ind.at_ind[a,b]
                disp_e = disp.at_disp[a,b]
                total_e = elst_e + exch_e + indu_e + disp_e
                outf.write("%3d %3d %12.8f %12.8f %12.8f %12.8f %12.8f \n" % (a+1,b+1,elst_e, exch_e, indu_e, disp_e, total_e) ) 
        
    if frag:
        generate_frag_output(filenames, mtp.at_elst, rep.at_exch, ind.at_ind, disp.at_disp)

    # for printing
    ret = {'elst':elst, 'exch':exch, 'indu':indu, 'disp':disp_en, 'total':elst+exch+indu+disp_en}
    return ret


def generate_frag_output(files, elst, exch, indu, disp):
    # get the files that define the fragments

    fa = files[0]
    fb = files[1]

    pref_a = fa.split('.xyz')[0]  
    pref_b = fb.split('.xyz')[0]  

    fa = pref_a + '-frag.dat'
    fb = pref_b + '-frag.dat'

    pref_a = pref_a.split('/')[-1]
    pref_b = pref_b.split('/')[-1]

    fa_frags = {}
    fb_frags = {}

    # load the fragments
    with open(fa, 'r') as fa_file:
        all_a = []
        for line in fa_file:
            line = line.split()
            atms = []
            for item in line[1:]:
                atms.append(item)
                all_a.append(item)
            fa_frags[line[0]] = atms
        fa_frags['All'] = all_a

    with open(fb, 'r') as fb_file:
        all_b = []
        for line in fb_file:
            line = line.split()
            atms = []
            for item in line[1:]:
                atms.append(int(item))
                all_b.append(item)
            fb_frags[line[0]] = atms
        fb_frags['All'] = all_b
    
    with open(pref_a + "-" + pref_b + "-frag.txt",'w') as ffile:
        for fa, atoms_a in fa_frags.items():
            for fb, atoms_b in fb_frags.items():
                elst_e = 0.0 
                exch_e = 0.0
                indu_e = 0.0
                disp_e = 0.0
                total_e = 0.0           
                for na in atoms_a:
                    for nb in atoms_b:
                        elst_a = elst[int(na)-1,int(nb)-1]  
                        exch_a = exch[int(na)-1,int(nb)-1]  
                        indu_a = indu[int(na)-1,int(nb)-1]  
                        disp_a = disp[int(na)-1,int(nb)-1]  

                        elst_e += elst_a
                        exch_e += exch_a
                        indu_e += indu_a
                        disp_e += disp_a

                        total_e += elst_a + exch_a + indu_a + disp_a

                         
                ffile.write("%s %s %12.8f %12.8f %12.8f %12.8f %12.8f \n" % (fa,fb,elst_e, exch_e, indu_e, disp_e, total_e) ) 
            
    

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
    logger.info("           MonomerA     |      MonomerB      |  Electrostatics |   Exchange   |   Induction   |   Dispersion  |   Total ")
    logger.info("    ----------------------------------------------------------------------------------------------------------------------") 

    with open(name + '.json','w') as out:
        json.dump(ret,out)

    with open(name + '.csv','w') as cout:
        cout.write("# Monomer A, Monomer B, Electrostatics, Exchange, Induction, Dispersion, Total (kcal/mol)")
        for k,val in ret.items():
            mona, monb, v = val
            logger.info("    %-20s %-20s%18.5f %14.5f %15.5f %15.5f %11.5f" % (mona,monb, v['elst'],v['exch'],v['indu'],v['disp'],v['total']))
            cout.write("\n%s,%s,%9.5f,%9.5f,%9.5f,%9.5f,%9.5f" % (mona,monb, v['elst'],v['exch'],v['indu'],v['disp'],v['total']))

def print_timings(timer):
    logger.info("")
    logger.info("    ~Component Timings")
    logger.info("    ~Electrostatics:  %10.3f s" % timer['elst'])
    logger.info("    ~Exchange      :  %10.3f s" % timer['exch'])
    logger.info("    ~Induction     :  %10.3f s" % timer['ind'])
    logger.info("    ~Dispersion    :  %10.3f s" % timer['disp'])

def main(inpt=None, files=None, ref=None, nproc=None, name=None, frag = None):
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
            mona =  filenames[0].split('/')[-1].split('.xyz')[0]
            monb = filenames[1].split('/')[-1].split('.xyz')[0]
            dname = mona + "-" + monb
        else: 
            mona =  filenames[0].split('/')[-1].split('.xyz')[0]
            monb = filenames[1].split('/')[-1].split('.xyz')[0]
            dname = filenames[0].split('/')[-2]
        en = get_energy(filenames, models, options, name, timer, frag)
        ret[dname] = [mona,monb,en]

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

    frag = False
    if args.frag != False:
        frag = True

    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(name + '.log')
    logger.addHandler(fh)

    main(args.input, args.files, args.ref, args.nproc, name, frag)
    end = time.time()

    logger.info("    ~CLIFF ran in {} s".format(end-start))

