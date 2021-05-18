#! /usr/bin/env python


"""
driver.py
Component-based Learned Intermolecular Force Field

Contains all functions intended for outermost API-layer

"""

import os


import numpy as np
import glob
import qcelemental as qcel

#import cliff
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

def mol_to_sys(mol, options):
    
    # If mol is a dimer, return two sys
    
    nfrag = len(mol.fragments)
    dimer_name = mol.name
    systems = []
    for frag in range(nfrag):
        fmol = mol.get_fragment(frag)
        sys = System(options)

        if nfrag == 1:
            sys.load_qcel_mol(fmol, name = dimer_name)
        else:
            f_name = dimer_name + "-" + str(frag)
            sys.load_qcel_mol(fmol, name = f_name)
    
        systems.append(sys)

    if nfrag == 1:
        return systems[0]
    else:
        return systems        
    

def load_dimer_xyz(xyz_file, units='angstrom'):
    '''
    Loads xyz file, where the separation between dimers is
    specified as the last field in the comment line
    
    Returns single qcel Molecule object
    '''

    lines = open(xyz_file,'r').readlines()

    natom = int(lines[0].strip())
    data = lines[1].strip()
    coords = lines[-natom:]

    na = int(data.split(',')[-1])
    nb = natom - na

    mol_name = xyz_file.split("/")[-1].split(".xyz")[0]
    blockA = f"0 1\n" + "".join(coords[:na])
    blockB = f"0 1\n" + "".join(coords[na:])
    dimer = blockA + "--\n" + blockB + f"no_com\nno_reorient\nunits {units}"
    dimer = qcel.models.Molecule.from_data(dimer, **{'name':mol_name})

    return dimer
    


def load_monomer_xyz(xyz_file, units='angstrom'):
    '''
    Loads single/multi xyz file of monomers
    Total charge is last field in comment line    

    Returns list of qcel Molecule objects
    '''
    

    molecules = []    

    lines = open(xyz_file,'r').readlines()
    atom_n = [int(line.split()[0]) for line in lines if (len(line.split()) == 1) and (len(line.split(',')) == 1)]
    charges = [int(line.split(',')[-1]) for line in lines if (len(line.split(',')) > 1)]
    coords = [line for line in lines if len(line.split()) == 4]


    n_prev = 0
    nmol = 0
    for z,n in zip(charges,atom_n):
        mol = f"{z} 1\n" + "".join(coords[n_prev:n_prev + n]) 
        mol += f"no_com\nno_reorient\nunits {units}"

        n_prev += n

        mol_name = xyz_file.split("/")[-1].split(".xyz")[0] + "-" + str(nmol)
        molecules.append(qcel.models.Molecule.from_data(mol,**{'name':mol_name}))
        nmol += 1

    return molecules
    

def predict_atomic_properties(mol, models):
    hirsh = models[0]
    adens = models[1]
    mtp_ml = models[2]

    hirsh.predict_mol(mol, force_predict=True)
    adens.predict_mol(mol, force_predict=True)
    mtp_ml.predict_mol(mol, force_predict=True)
 
    return mol    
    
def save_atomic_properties(mol,path):
    # Only do properties that have been computed
    if len(mol.hirshfeld_ratios) > 0:
        sfile = mol.name + "-h.npy"
        np.save(path + "/" + sfile, mol.hirshfeld_ratios) 
    if len(mol.valence_widths) > 0:
        sfile = mol.name + "-vw.npy"
        np.save(path + "/" + sfile, mol.valence_widths) 
    if len(mol.multipoles) > 0:
        sfile = mol.name + "-mtp.npy"
        np.save(path + "/" + sfile, mol.multipoles) 

def load_atomic_properties(mol,path):
    mol.hirshfeld_ratios = np.load(path + "/" + mol.name + "-h.npy")  
    mol.valence_widths = np.load(path + "/" + mol.name + "-vw.npy")  
    mol.multipoles = np.load(path + "/" + mol.name + "-mtp.npy")  
        
    return mol

def predict_from_dimers(dimers, load_path=None, return_pairs=False, infile=None, options=None):
    '''
    Compute energy components from a list of dimers.
    Uses all default options, turns off logging

    Parameters
    ----------
    dimers: qcel Molecule class or list of Molecule class
    return_pairs: bool to control returning of full atom-pairwise decomposition
 
    '''

    if isinstance(dimers,list):
        d_list = dimers
    else:
        d_list = [dimers]     


    # load options (all defaults) and get the KRR models
    if options is None:
        if infile is None:
            options = Options()
        else:
            options = Options(config_file=infile)

    if load_path is None:
        models = load_krr_models(options) 
   
    energies = []

    for dimer in d_list:

        mon_a, mon_b = mol_to_sys(dimer, options)

        if load_path is None:
            mon_a = predict_atomic_properties(mon_a,models)
            mon_b = predict_atomic_properties(mon_b,models)
        else:
            mon_a = load_atomic_properties(mon_a,load_path)  
            mon_b = load_atomic_properties(mon_b,load_path)  

        en = energy_kernel(mon_a, mon_b, options) 
        #try:
        #    en = energy_kernel(mon_a, mon_b, options) 
        #except:
        #    en = "Error"

        energies.append(en)

    return np.asarray(energies) 
    
def predict_from_monomer_list(monomer_a, monomer_b, load_path=None, return_pairs=False,infile=None, options=None):
    '''
    Compute energy components from two lists of monomers
    Uses default options, places mon_a in outer loop 
    monomer list to outer loop

    Parameters
    ----------

    mon_a: list of (or single) qcel Molecules
    mon_b: list of (or single) qcel Molecules

    return_pairs: bool to control returning of full atom-pairwise decomposition
    '''

    if isinstance(monomer_a,list):
        mon_a_list = monomer_a
    else:
        mon_a_list = [monomer_a]

    if isinstance(monomer_b,list):
        mon_b_list = monomer_b
    else:
        mon_b_list = [monomer_b]

    if options is None:
        if infile is None:
            options = Options()
        else:
            options = Options(config_file=infile)

    if load_path is None:
        models = load_krr_models(options) 

    energies = []
    for A in mon_a_list:
        mon_a = mol_to_sys(A, options)
        
        if load_path is None:
            mon_a = predict_atomic_properties(mon_a,models)
        else:
            mon_a = load_atomic_properties(mon_a,load_path)  
            

        for B in mon_b_list:
            mon_b = mol_to_sys(B, options)

            if load_path is None:
                mon_b = predict_atomic_properties(mon_b,models)
            else:
                mon_b = load_atomic_properties(mon_b,load_path)  

            try:
                en = energy_kernel(mon_a, mon_b, options) 
            except:
                en = "Error"

            energies.append(en)

    return energies

def energy_kernel(mon_a, mon_b, options):
    
    #defines cell parameters for grid computations
    cell = Cell.lattice_parameters(100., 100., 100.)

    #initializes relevant classes with monomer A
    mtp = Electrostatics(options,mon_a, cell)
    ind = InductionCalc(options,mon_a, cell)
    rep = Repulsion(options, mon_a, cell)
    disp = Dispersion(options, mon_a, cell) 
    
    #adds monomer B
    mtp.add_system(mon_b)
    ind.add_system(mon_b)
    rep.add_system(mon_b)
    disp.add_system(mon_b)
    
    #computes electrostatic, induction and exchange energies
    elst_n = mtp.mtp_energy()
    indu_n = ind.polarization_energy()
    exch_n = rep.compute_repulsion()
    disp_en = disp.compute_dispersion()  

    return np.array([elst_n,exch_n,indu_n,disp_en]) 

def load_krr_models(options):
    hirsh = Hirshfeld(options,None)
    adens = AtomicDensity(options,None)
    mtp = Multipole(options,None)

    hirsh.load_ml()
    adens.load_ml()
    mtp.load_ml()

    return [hirsh,adens,mtp]


