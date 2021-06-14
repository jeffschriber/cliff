#! /usr/bin/env python

"""
train.py
Component-based Learned Intermolecular Force Field

Functions for fitting global parameters
"""

import math
import numpy as np 
import scipy.optimize as opt
import json
import operator
import glob
import pprint

import cliff
from cliff.helpers.options import Options


def get_energy(params, pathname, gamma, ref):


    # grab parameters
    elst_params = params[:17]
    exch_params = params[17:34]
    indu_params = params[34:52]
    disp_params = params[52:]
    
    elst_param_dict = {
    'Cl' : abs(elst_params[0] )  , 
    'F'  : abs(elst_params[1] )  , 
    'S1' : abs(elst_params[2] )  , 
    'S2' : abs(elst_params[3] )  , 
    'HS' : abs(elst_params[4] )  , 
    'HC' : abs(elst_params[5] )  ,
    'HN' : abs(elst_params[6] )  , 
    'HO' : abs(elst_params[7] )  , 
    'C4' : abs(elst_params[8] )  , 
    'C3' : abs(elst_params[9] )  , 
    'C2' : abs(elst_params[10])   , 
    'N3' : abs(elst_params[11])   ,
    'N2' : abs(elst_params[12])   , 
    'N1' : abs(elst_params[13])   , 
    'O1' : abs(elst_params[14])   , 
    'O2' : abs(elst_params[15]) ,
    'Br' : abs(elst_params[16])}  

    exch_param_dict = {
    'Cl' : abs(exch_params[0] )  , 
    'F'  : abs(exch_params[1] )  , 
    'S1' : abs(exch_params[2] )  , 
    'S2' : abs(exch_params[3] )  , 
    'HS' : abs(exch_params[4] )  , 
    'HC' : abs(exch_params[5] )  ,
    'HN' : abs(exch_params[6] )  , 
    'HO' : abs(exch_params[7] )  , 
    'C4' : abs(exch_params[8] )  , 
    'C3' : abs(exch_params[9] )  , 
    'C2' : abs(exch_params[10])   , 
    'N3' : abs(exch_params[11])   ,
    'N2' : abs(exch_params[12])   , 
    'N1' : abs(exch_params[13])   , 
    'O1' : abs(exch_params[14])   , 
    'O2' : abs(exch_params[15]) ,
    'Br' : abs(exch_params[16])}  

    indu_param_dict = {
    'Cl' : abs(indu_params[1]),
    'F'  : abs(indu_params[2]), 
    'S1' : abs(indu_params[3]),
    'S2' : abs(indu_params[4]),
    'HS' : abs(indu_params[5]),
    'HC' : abs(indu_params[6]),
    'HN' : abs(indu_params[7]),
    'HO' : abs(indu_params[8]),
    'C4' : abs(indu_params[9]),
    'C3' : abs(indu_params[10]),
    'C2' : abs(indu_params[11]), 
    'N3' : abs(indu_params[12]), 
    'N2' : abs(indu_params[13]), 
    'N1' : abs(indu_params[14]), 
    'O1' : abs(indu_params[15]), 
    'O2' : abs(indu_params[16]), 
    'Br' : abs(indu_params[17]) }

    disp_param_dict = {
    'Cl' : abs(disp_params[0] )  , 
    'F'  : abs(disp_params[1] )  , 
    'S1' : abs(disp_params[2] )  , 
    'S2' : abs(disp_params[3] )  , 
    'HS' : abs(disp_params[4] )  , 
    'HC' : abs(disp_params[5] )  ,
    'HN' : abs(disp_params[6] )  , 
    'HO' : abs(disp_params[7] )  , 
    'C4' : abs(disp_params[8] )  , 
    'C3' : abs(disp_params[9] )  , 
    'C2' : abs(disp_params[10])   , 
    'N3' : abs(disp_params[11])   ,
    'N2' : abs(disp_params[12])   , 
    'N1' : abs(disp_params[13])   , 
    'O1' : abs(disp_params[14])   , 
    'O2' : abs(disp_params[15]) ,
    'Br' : abs(disp_params[16])}  
    
    print("Parameters: ")
    print("Elst")
    pprint.pprint(elst_param_dict)
    print("Exch")
    pprint.pprint(exch_param_dict)
    print("Indu")
    print(f"Smearing coefficient: {abs(indu_params[0])}")
    pprint.pprint(indu_param_dict)
    print("Disp")
    pprint.pprint(disp_param_dict)

    #set globals
    options = Options('config.ini')
    options.set_damping_exponents(elst_param_dict)
    options.set_indu_smearing_coeff(abs(indu_params[0]))
    options.set_induction_sr_params(indu_param_dict)
    options.set_exchange_int_params(exch_param_dict)
    options.set_disp_coeffs(disp_param_dict)
    
    dimer_xyz = sorted(glob.glob(pathname + "/*.xyz"))
    dimers = [cliff.load_dimer_xyz(f) for f in dimer_xyz]
    energies = cliff.predict_from_dimers(dimers)

    total_e = energies[:,0]
    elst_e = energies[:,1]
    exch_e = energies[:,2]
    indu_e = energies[:,3]
    disp_e = energies[:,4]

    total_r = ref[:,0]
    elst_r  = ref[:,1]
    exch_r  = ref[:,2]
    indu_r  = ref[:,3]
    disp_r  = ref[:,4]

    total_err = total_r - total_e
    elst_err = elst_r - elst_e
    exch_err = exch_r - exch_e
    indu_err = indu_r - indu_e
    disp_err = disp_r - disp_e

    mse_total = np.average(np.square(total_err))
    mse_elst = np.average(np.square(elst_err))
    mse_exch = np.average(np.square(exch_err))
    mse_indu = np.average(np.square(indu_err))
    mse_disp = np.average(np.square(disp_err))

    print("MSEs (kcal/mol):")
    print(f"Elst    {mse_elst}")
    print(f"Exch    {mse_exch}")
    print(f"Indu    {mse_indu}")
    print(f"Disp    {mse_disp}")


    ret_val = gamma * (mse_elst + mse_exch + mse_indu + mse_disp) + (1.0-gamma) * mse_total
    print(f"Multi-target metric (gamma = {gamma}): {ret_val}")
    return ret_val

def fit_global_parameters(pathname, ref_dict, initial_guess=None,gamma=0.4, method='bfgs'):

"""
Fits global parameters used in CLIFF


Parameters
----------

pathname : :class: `str`
    Path to the dimer xyz files. The comment (second) line in the xyz needs to specify
    the number of atoms in the first monomer.
ref_dict : :class: `dict`
    Dictionary containing the reference energies for fitting. The keys to the dictionary
    need to be the extension-less dimer filenames, and the value is a numpy array of
    energies: [total, elst, exch. indu].
initial_guess : :class: `~numpy.ndarray`
    Numpy array of initial guess parameters (69 total). See the `get_energy` function
    for how initial parameters should be ordered. Default is 1.0 for all parameters
gamma : :class: `float`
    Gamma parameter between 0 and 1.0 that controls the degree to which the total 
    energy influences fitting.
method: :class: `str`
    Algorithm used for minimization. See scipy.minimize documentation for all options

"""

    atoms = ['Cl', 'F', 'Br', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2', 'N3', 'N2', 'N1', 'O1', 'O2']
    
    if initial_guess is not None:
        initial_guess = initial_guess.flatten()
        if len(initial_guess) != 69:
            raise Exception("Initial guess has the wrong dimension")
    else:
        initial_guess = np.ones(70)

    ref = []
    dimer_xyz = sorted(glob.glob(pathname + "/*.xyz"))
    for dimer in dimer_xyz:
        key = dimer.split('/')[-1].split('.xyz')[0]
        ref.append(ref_dict[key])
    ref = np.asarray(ref)
    res = opt.minimize(get_energy, initial_guess, method=method, args=(pathname, gamma,ref))
    print(res)
    return res
