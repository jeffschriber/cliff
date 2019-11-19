"""
cliff.py
Component-based Learned Intermolecular Force Field

Handles the primary functions
"""
#! /usr/bin/env/ python

# import relevant modules 
#import cliff
#from cliff.helpers.calculator import Calculator
#from cliff.helpers.cell import Cell
#from cliff.helpers.system import System
#from cliff.atomic_properties.hirshfeld import Hirshfeld
#from cliff.atomic_properties.multipole_ml_bset import MultipoleMLBSet
#from cliff.components.cp_multipoles import CPMultipoleCalc
#from cliff.components.induction_calc import InductionCalc
#from cliff.components.repulsion import Repulsion
#from cliff.components.dispersion import Dispersion
#import numpy as np
#from functools import reduce
#import configparser
#import operator
#
#def get_energy(filename):
#    np.set_printoptions(precision=4, suppress=True, linewidth=100)
#    
#    #1. Initialize relevant variables
#    
#    #loads parameters contained on the config.init file
#    calc = Calculator() 
#    
#    #defines cell parameters for grid computations
#    cell = Cell.lattice_parameters(100., 100., 100.)
#    
#    #initializes Hirshfeld class
#    hirsh = Hirshfeld(calc) 
#    
#    #load KRR model for Hirshfeld as specified in the config.init file
#    hirsh.load_ml() 
#    
#    #load multipoles with aSLATM representation
#    mtp_ml  = MultipoleMLBSet(calc, descriptor="slatm") 
#    
#    #loads monomer geometries
#    mols = []
#    xyzs = [] 
#    for xyz in filename:
#        #computes descriptor for each monomer   
#        mols.append(System(xyz))
#        xyzs.append(xyz)
#    
#    for mol,xyz in zip(mols,xyzs):
#        #predicts monomer multipole moments for each monomer
#        mtp_ml.predict_mol(mol)
#        #predicts Hirshfeld ratios usin KRR
#        hirsh.predict_mol(mol,"krr")
#    
#    #initializes relevant classes with monomer A
#    mtp = CPMultipoleCalc(mols[0], cell)
#    ind = InductionCalc(mols[0], cell)
#    rep = Repulsion(mols[0], cell)
#    
#    #adds monomer B
#    for mol in mols[1:]:
#        mtp.add_system(mol)
#        ind.add_system(mol)
#        rep.add_system(mol)
#    
#    #computes electrostatic, induction and exchange energies
#    elst = mtp.mtp_energy()
#    indu = ind.polarization_energy()
#    exch = rep.compute_repulsion("slater_mbis")
#    
#    #creat dimer
#    dimer = reduce(operator.add, mols)
#
#    #compute hirshfeld_ratios in the dimer basis 
#    hirsh.predict_mol(dimer, "krr")   
#    
#    #use Hirshfeld ratios in the computation of dispersion energy
#    #as disp = E_dim - E_monA - E_monB
#    disp = 0.0
#    for i, mol in enumerate([dimer] + mols):
#        fac = 1.0 if i == 0 else -1.0
#        #initialize Dispersion class 
#        mbd = Dispersion(mol, cell)
#        #compute C6 coefficients
#        mbd.compute_csix()
#        #compute anisotropic characteristic frequencies
#        mbd.compute_freq_scaled_anisotropic()
#        #execute MBD protocol
#        mbd.mbd_protocol(None,None,None)
#        disp += fac * mbd.energy
#     
#    # for printing
#    return elst, exch, indu, disp,  elst+exch+indu+disp
#

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

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
