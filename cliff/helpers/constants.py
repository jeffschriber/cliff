#!/usr/bin/env python
#
# List of global parameters and constants.
#
# Tristan Bereau (2017)

# Atomic units to kcal/mol
au2kcalmol    = 627.5095
# angstrom to bohr
a2b           = 1.8897268
# Bohr to Angstrom
b2a           = 0.529177
# Multipole conversion: Hartree and bohr to kcal/mol (332.064)
hbohr2kcalmol = 332.063595

# Some mathematical constants
sqrt_3      = 1.732050808

# Degrees to radian
deg2rad     = 0.017453292519943295

# au to debye
au2debye    = 2.541746231
debye2au    = 0.393430307

atomic_number = {
  'H'  : 1,
  'B'  : 5,
  'C'  : 6,
  'N'  : 7,
  'O'  : 8,
  'F'  : 9,
  'P'  :15,
  'S'  :16,
  'Cl' :17,
  'Br' :35,
  'I'  :53,
}

Z_val = {
    'H' : 1,
    'B' : 3,
    'C' : 4,
    'N' : 5,
    'O' : 6,
    'F' : 7,
    'P' : 5,
    'S' : 6,
    'Cl': 7,
    'Br': 7,
    'I' : 7,
}

atom_row = {
    'H' : 0,
    'B' : 1,
    'C' : 1,
    'N' : 1,
    'O' : 1,
    'F' : 1,
    'P' : 2,
    'S' : 2,
    'Cl': 2,
    'Br': 3,
    'I' : 4,
}

pol_free = {
  'H' :  4.50,
  'He':  1.38,
  'C' : 12.00,
  'N' :  7.40,
  'O' :  5.40,
  'F' :  3.80,
  'Ne':  2.67,
  'Si': 37.00,
  'P' : 25.00,
  'S' : 19.60,
  'Cl': 15.00,
  'Ar': 11.10,
  'Br': 20.00,
  'Kr': 16.80,
  'I' : 35.00,
}

csix_free  = {
  'H' :  6.50,
  'He':  1.46,
  'C' : 46.60,
  'N' : 24.20,
  'O' : 15.60,
  'F' :  9.52,
  'Ne':  6.38,
  'Si': 305.0,
  'P' : 185.0,
  'S' : 134.0,
  'Cl': 94.60,
  'Ar': 64.30,
  'Br': 162.0,
  'Kr': 130.0,
  'I' : 385.0,
}

rad_free = {
  'H' : 3.10,
  'He': 2.65,
  'C' : 3.59,
  'N' : 3.34,
  'O' : 3.19,
  'F' : 3.04,
  'Ne': 2.91,
  'Si': 4.20,
  'P' : 4.01,
  'S' : 3.86,
  'Cl': 3.71,
  'Ar': 3.55,
  'Br': 3.93,
  'Kr': 3.82,
  'I' : 4.39,
}

atomic_r2 = {
 'C': 14.046740332406095,
'Cl': 27.651441841502898,
 'F': 10.601441421661603,
 'H': 3.1589186627858976,
 'N': 12.419553635522538,
 'O': 11.566513469042524,
 'S': 29.17364957903178,
'Br':40.3810331885329
}


atomic_r4 = {
  'C':  109.39192409419,
 'Cl':  187.29318856137,
  'F':  41.707556384507,
  'H':  26.818416656404,
  'N':  71.475971390365,
  'O':  56.180301267329,
  'S': 239.014016575349,
 'Br': 288.05982633399606
}

atomic_weight = {
  'H'  : 1.008,
  'B'  : 10.81,
  'C'  : 12.01,
  'N'  : 14.01,
  'O'  : 16.00,
  'F'  : 19.00,
  'P'  : 31.00,
  'S'  : 32.06,
  'Cl' : 35.45,
  'Br' : 79.90,
  'I'  : 126.90,
}

# Covalent radii
cov_rad = {
    'H' : 0.31,
    'C' : 0.70,
    'N' : 0.71,
    'O' : 0.66,
    'F' : 0.57,
    'P' : 1.07,
    'S' : 1.05,
    'Cl': 1.02,
    'Br': 1.20,
    'I' : 1.39,
}

# Hbond interaction: Strength k_hbnd (according to Grimme)
k_hbnd = {
    'N' : 0.8,
    'O' : 0.3,
    'F' : 0.1,
    'P' : 2.0,
    'S' : 2.0,
    'Cl': 2.0,
    'Br': 2.0,
    'I' : 2.0,
}

# Map multipole coefficient to index
map_mtp_coeff = {
  'Q10'  : 2,
  'Q11c' : 0,
  'Q11s' : 1,
  'Q20'  : 0,
  'Q21c' : 1,
  'Q21s' : 2,
  'Q22c' : 3,
  'Q22s' : 4,
}

# Atoms for bag of bonds (correctly ordered)
bob_atoms = ['C','H','O','S','N','Br','Cl','F','I','P','B']

# Charge penetration
# Follows Wang et al. JCTC (2017) DOI: 10.1021/acs.jctc.5b00267
# effective core charge
cp_Z = {
  'H'  : 1,
  'B'  : 3,
  'C'  : 4,
  'N'  : 5,
  'O'  : 6,
  'F'  : 7,
  'P'  : 5,
  'S'  : 6,
  'Cl' : 7,
  'Br' : 7,
}

# valence-alpha set [Ang^-1]
cp_alpha = {
  'H'  : 2.0,
  'C'  : 4.0,
  'N'  : 5.0,
  'O'  : 6.0,
  'F'  : 7.0,
  'P'  : 5.0,
  'S'  : 6.0,
  'Cl' : 7.0,
  'Br' : 7.0,
}

# Free-atom valence widths [Bohr^-1]
val_width_free = {
    'H': 0.5094,
    'C': 0.5242,
    'N': 0.4415,
    'O': 0.3882,
}

# Free-atom valence charges
val_charge_free = {
    'H': -1.00000061,
    'C': -4.31910903,
    'N': -5.35306426,
    'O': -6.36289409,
}

ml_metric = {
    'gaussian': 'euclidean',
    'laplacian': 'cityblock'
}

ml_prefactor = {
    'gaussian': 2.0,
    'laplacian': 1.0
}

ml_power = {
    'gaussian': 2,
    'laplacian': 1
}


# This is how IPML re-shuffles charge when correcting
# for non-integer total charge. Instead of these parameters.
# we'll use atomic weights
#ml_chg_correct_error = {
#    'H': 1.,
#    'C': 1.,
#    'N': 1.,
#    'O': 1.,
#    'S': 1., # NOT SURE WHAT VALUE
#}
#ml_chg_correct_error = {
#    'H': 1.,
#    'C': 3.2,
#    'N': 1.7,
#    'O': 1.,
#    'S': 1., # NOT SURE WHAT VALUE
#}
# based on mae
#ml_chg_correct_error = {
#    'H': 0.144236,
#    'C': 0.362682,
#    'N': 0.463021,
#    'O': 0.290041,
#    'S': 0.741042, 
#}
ml_chg_correct_error = {
    'H'  : 1.0,
    'C'  : 2.5145,
    'N'  : 3.2101,
    'O'  : 2.0108,
    'S'  : 5.108, 
    'Cl' : 2.703,
    'F'  : 1.138, 
    'Br'  : 1.138 
}

## Default damping parameters 
##for CP-corrected electrostatics
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

# Default short-range induction parameters
indu_sr_params = {
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
'O2'  :  0.841367
}

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

disp_coeffs = {
'Cl1' : 0.98270815 , 
'F1'  : 0.07118267 , 
'S1'  : 0.97732995 , 
'S2'  : 0.95325066 ,
'HS'  : 0.21635708 ,
'HC'  : 0.3154296  , 
'HN'  : 0.46146509 , 
'HO'  : 0.48129702 , 
'C4'  : 0.76002737 , 
'C3'  : 0.57735844 , 
'C2'  : 0.48336277 ,
'N3'  : 0.13779041 , 
'N2'  : 0.7207346  , 
'N1'  : 0.70831499 , 
'O1'  : 0.80784731 , 
'O2'  : 0.41045399 
}


# Default smearing coefficient for induction (Thole model)
indu_smearing_coeff = 0.5478502 


