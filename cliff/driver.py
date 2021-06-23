#! /usr/bin/env python


"""
Driver functions contain the main functions for energy computation,
in addition to related functions needed for predicting atomic properties.

"""

import os


using_apnet = True 
try:
    import apnet
except:
    using_apnet = False

import time
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

def set_nthread(nthread):
    """
    Sets the number of threads used by the entire CLIFF module.
    Increasing threads can only impact any threaded numpy routines.

    Parameters
    ----------
    nthread : :class: `int`
        Number of threads.


    Returns
    -------
        void
    """

    os.environ["OMP_NUM_THREADS"]    = str(nthread) 
    os.environ["MKL_NUM_THREADS"]    = str(nthread)
    os.environ["NUMEXPR_NUM_THREADS"]= str(nthread)

def mol_to_sys(mol, options):
    """
    Converts a qcelemental Molecule into a System object used by CLIFF.
    The input molecule object can specify either a dimer or a monomer, 
    though it has to have a name (mol.name) in either case.

    Parameters
    ----------
    mol : :class: `~qcel.models.Molecule`
        Input QCElemental molecule.
    options : :class: `~cliff.helpers.Options`
        Options object

    Returns
    -------
    systems : :class:`cliff.System`
        Cliff System object 
    """
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
    Loads a dimer xyz file, where the separation between dimers is
    specified as the last field in the second line in the file.
    
    Parameters
    ----------
    xyz_file : :class:`str`
        Filename of the dimer xyz file. See main docs for how to format a dimer xyz.
    units : :class:`str`:
        Units of the input coordinates, default to Angstrom. See QCElemental documentation
        for all available units.

    Returns
    -------
    dimer : :class:`~qcelemental.molecule`
        QCElemental Molecule object. The returned Molecule object has two defined fragemens, one for
        each monomer. 
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
    Loads single/multi xyz file of monomers.
    Total charge is last field in comment line.    

    Function returns list of qcel Molecule objects.

    Parameters
    ----------

    xyz_file : :class: `string`
        Single xyz file containing one or more monomer coordinates. Each monomer
        is in the usual xyz format, with the exception that here we reqiure the 
        second line to specify the total charge as the final field in a comma-separated list
        with potentially other information before.
    units : :class:`str`:
        Units of the input coordinates, default to Angstrom. See QCElemental documentation
        for all available units.
    
    Returns
    -------
        molecules : list of :class:`qcelemental.molecule` objects
            A List of QCElemental molecule objects, each corresponding to a monomer in the input monomer xyz
            file.  

    '''
    

    molecules = []    

    lines = open(xyz_file,'r').readlines()
    atom_n = [int(line.split()[0]) for line in lines if (len(line.split()) == 1) and (len(line.split(',')) == 1)]
    charges = [int(line.split(',')[-1]) for line in lines if (len(line.split(',')) > 1)]
    coords = [line for line in lines if len(line.split()) == 4]

    if len(charges) == 0:
        raise Exception("Must specify total charge in xyz file (last field in comment line of xyz)")

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
    """
    Predicts the atomic properties for an input System
    Returns System with Hirshfeld, atomic density, and multipole.

    Parameters
    ----------
    mol : :class: `~cliff.helpers.System`
        Input System
    models : list of :class:`~cliff.atomic_properties.Hirshfeld` ,`~cliff.atomic_properties.AtomicDensity`, and `~cliff.atomic_properties.Multipole`
        List of dimension (3,) in the exact order: [Hirshfeld, AtomicDensity, Multipole].
    """

    hirsh = models[0]
    adens = models[1]
    mtp_ml = models[2]

    hirsh.predict_mol(mol, force_predict=True)
    adens.predict_mol(mol, force_predict=True)
    mtp_ml.predict_mol(mol, force_predict=True)
 
    return mol    
    
def save_atomic_properties(mol,path):
    """
    Saves atomic properties to a .npy file in a specified location.
    The name of the file is determined by the name of the original
    qcelemental Molecule object, with the suffix -h.npy, -vw.npy, or
    -mtp.npy for a given atomic property.

    Parameters
    ----------
    mol : :class: `~cliff.helpers.System`
        System object with one or more atomic property fields already computed.
    path : :class: `str`
        Path to the destination where atomic property files are to be stored.
        Note that this path needs to exist, and it cannot be used to specify a filename. 
    """
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
    """
    Loads Hirshfeld ratios, valence widths, and multipoles
    from files and stores them in the input System object.
    The System object finds the appropriate 
    """

    mol.hirshfeld_ratios = np.load(path + "/" + mol.name + "-h.npy")  
    mol.valence_widths = np.load(path + "/" + mol.name + "-vw.npy")  
    mol.multipoles = np.load(path + "/" + mol.name + "-mtp.npy")  
        
    return mol

def predict_from_dimers(dimers, ml_type='KRR', load_path=None, return_pairs=False, infile=None, options=None):
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

   
    energies = []
    mon_a_list = []
    mon_b_list = []

    s = time.time()
    if ml_type.upper() == "KRR":
        # get atomic properties
        if load_path is None:
            models = load_krr_models(options) 

        # get the monomers
        for dimer in d_list:
            try:
                mon_a, mon_b = mol_to_sys(dimer, options)
                if load_path is None:
                    mon_a = predict_atomic_properties(mon_a,models)
                    mon_b = predict_atomic_properties(mon_b,models)
                else:
                    mon_a = load_atomic_properties(mon_a,load_path)  
                    mon_b = load_atomic_properties(mon_b,load_path)  
                mon_a_list.append(mon_a)    
                mon_b_list.append(mon_b)    
            except:
                mon_a_list.append(None)    
                mon_b_list.append(None)    
    elif (ml_type.upper() == "NN") and using_apnet:
        ma_s = []
        mb_s = []
        for dimer in d_list:
            ma_s.append(dimer.get_fragment(0))
            mb_s.append(dimer.get_fragment(1))

        model_path = os.path.dirname(os.path.realpath(__file__))
        model_path += '/models/apnet/cliff_pbe0atz.h5'

        s1 = time.time()
        ma_props = apnet.predict_cliff_properties(ma_s, model_path)
        mb_props = apnet.predict_cliff_properties(mb_s, model_path)
        f1 = time.time()
        print(f"apnet time: {f1-s1} s")

        for nd, dimer in enumerate(d_list):        
            mon_a, mon_b = mol_to_sys(dimer, options)

            mon_a.set_properties(ma_props[nd])
            mon_b.set_properties(mb_props[nd])

            mon_a_list.append(mon_a)    
            mon_b_list.append(mon_b)    
    elif (ml_type.upper() == "NN") and not using_apnet:
        raise Exception(f"ML type {ml_type} requested, but APNET not found!")
    else:
        raise Exception(f"ML type {ml_type} not understood!") 

    f = time.time()
    print(f"Time spent predicting atomic properties: {f-s} s")
        
    for ma, mb in zip(mon_a_list,mon_b_list):
        try:
            en = energy_kernel(ma, mb, options, return_pairs=return_pairs) 
        except:
            en = None

        energies.append(en)

    return energies
    
def predict_from_monomer_list(monomer_a, monomer_b,ml_type='KRR', load_path=None, return_pairs=False,infile=None, options=None):
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

    mon_a_sys = []
    mon_b_sys = []

    if ml_type.upper() == "KRR":
 
        if load_path is None:
            models = load_krr_models(options) 

        for A in mon_a_list:
            try:
                mon_a = mol_to_sys(A, options)
                
                if load_path is None:
                    mon_a = predict_atomic_properties(mon_a,models)
                else:
                    mon_a = load_atomic_properties(mon_a,load_path)  
                mon_a_sys.append(mon_a)
            except:
                mon_a_sys.append(None)
                

        for B in mon_b_list:
            try:
                mon_b = mol_to_sys(B, options)

                if load_path is None:
                    mon_b = predict_atomic_properties(mon_b,models)
                else:
                    mon_b = load_atomic_properties(mon_b,load_path)  
                mon_b_sys.append(mon_b)
            except:
                mon_b_sys.append(None)

    elif (ml_type.upper() == "NN") and (using_apnet):
        model_path = os.path.dirname(os.path.realpath(__file__))
        model_path += '/models/apnet/cliff_pbe0atz.h5'

        ma_props = apnet.predict_cliff_properties(mon_a_list, model_path)
        mb_props = apnet.predict_cliff_properties(mon_b_list, model_path)

        for nA, A in enumerate(mon_a_list):
            mon_a = mol_to_sys(A, options)
            mon_a.set_properties(ma_props[nA])
            mon_a_sys.append(mon_a)
            
        for nB, B in enumerate(mon_b_list):
            mon_b = mol_to_sys(B, options)
            mon_b.set_properties(mb_props[nB])
            mon_b_sys.append(mon_b)
    elif (ml_type.upper() == "NN") and not using_apnet:
        raise Exception(f"ML type {ml_type} requested, but APNET not found!")
    else:
        raise Exception(f"ML type {ml_type} not understood!") 


    energies = []
    for mon_a in mon_a_sys:
        for mon_b in mon_b_sys:
            try:
                en = energy_kernel(mon_a, mon_b, options, return_pairs=return_pairs) 
            except:
                en = None

            energies.append(en)

    return energies

def energy_kernel(mon_a, mon_b, options, return_pairs=False):
    
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
    total = elst_n + indu_n + exch_n + disp_en


    if return_pairs:
        elst_p = mtp.at_elst
        exch_p = rep.at_exch
        indu_p = ind.at_ind
        disp_p = disp.at_disp

        total_p = elst_p + exch_p + indu_p + disp_p

        ret_array = np.array([total_p,elst_p,exch_p,indu_p,disp_p])
    else:
        ret_array = np.array([total,elst_n,exch_n,indu_n,disp_en])

    return ret_array 

def load_krr_models(options):
    hirsh = Hirshfeld(options,None)
    adens = AtomicDensity(options,None)
    mtp = Multipole(options,None)

    hirsh.load_ml()
    adens.load_ml()
    mtp.load_ml()

    return [hirsh,adens,mtp]


